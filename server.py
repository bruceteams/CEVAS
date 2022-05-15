# encoding: utf-8
from concurrent import futures
import logging

import grpc,os,sys
import helloworld_pb2
import helloworld_pb2_grpc
import time,queue,threading
from PIL import Image
from io import BytesIO
import numpy as np
import torch,pickle
# import torch_model
import torchvision.transforms as T
import config_our.cluster_config as config
from config_our.cluster_config import Sharing_Object, Task,Logger,judge_blocksof_box
from tool_server import *
from tool_client import inter
from cal_iou import compute_iou

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
# prepare model: yolo v5
model_n = torch.hub.load('.cache/master', 'yolov5n', source='local')
model_n.eval()
model_s = torch.hub.load('.cache/master', 'yolov5s', source='local')
model_s.eval()
model_m = torch.hub.load('.cache/master', 'yolov5m', source='local')
model_m.eval()
model_list = [model_n,model_s,model_m]

local_role_ = "RSU-1"
task_queue = queue.Queue()  
request_queue = queue.Queue()
iou_queue = queue.Queue()   
sharing_object_list = []
compare_box_thre = 0.7    
update_dis = 40    
sharing_confi = 0.4   
area_threshold = 120*120  
start_time = {}
inference_time = []
camera_iou = {"c1":0.11,"c2":0.11,"c3":0.11,"c4":0.11}   
def sharing_object_manager():
    while True:
        task = task_queue.get()
        global sharing_object_list

        if len(task.sharing_box) != 0:
            np_result = np.append(task.result,task.sharing_box,axis=0)  
        else:
            np_result = task.result

        camera_role = task.task_id[:2]  # 'c3/ 1'
        num = int(task.task_id[4:])
        remove_list = []
        for i in range(len(np_result)):
            pred_item = np_result[i]
            for object in sharing_object_list:
                if object.num - num > update_dis:   
                    object.state = "stop"
                if camera_role in object.camera_list:   
                    box = pred_item[:4]
                    index = object.camera_list.index(camera_role)
                    iou = compute_iou(box,object.box[index])
                    if iou > compare_box_thre:
                        log.logger.debug("update %s based on sharing box %s iou %s",object.box[index],box,iou)
                        object.box[index] = box
                        # index = np.where(np_result == pred_item)[0][0]
                        remove_list.append(i)
        np_result = np.delete(np_result,remove_list,axis=0)  
        
        start = time.time()
        img = task.data
        config_region = config.region_list[camera_role]
        overlap_region = config_region['overlap']
        label_list = ['car','truck','bus']

        duplicate_object = []  
        for pred_item in np_result: 
            box = pred_item[:4]
            area = (int(box[3])-int(box[1])) * (int(box[2])-int(box[0]))
            blocks = judge_blocksof_box(box)
            l1 = inter(overlap_region,blocks)
            if l1 and (float(pred_item[4]) > sharing_confi) and (pred_item[-1] in label_list) and (area > area_threshold): 
                log.logger.debug("sharing object, id %s",task.task_id)
                color = extract_color(img,box)
                features = extract_feature(img,box) 
                matching = False  
                matching_object = None
                for object in sharing_object_list:
                    if object.state == "stop":  
                        continue
                    f_match = compare_existing_object(object,color,features)
                    if f_match:
                        matching = True
                        matching_object = object
                        break
                if not matching: 
                    new_object = Sharing_Object(pred_item[-1],pred_item[-2],[list(pred_item[:4])],[camera_role],color=color,features=features,num=num)
                    sharing_object_list.append(new_object)
                    log.logger.debug("create the new object task id %s",task.task_id)
                    object = img.crop(box)
                    num_p = len(sharing_object_list)
                    object.save("./outputs/sharing/"+str(num_p)+".jpg")
                else:  
                    if camera_role in matching_object.camera_list:  
                        index = matching_object.camera_list.index(camera_role)
                        matching_object.box[index] = list(pred_item[:4])
                        log.logger.debug("update box,id %s",task.task_id)
                    else:   
                        matching_object.box.append(list(pred_item[:4]))
                        matching_object.camera_list.append(camera_role)
                        duplicate_object.append(matching_object)
                    log.logger.debug("match to exist object id %s, camera list %s",task.task_id,matching_object.camera_list)
        log.logger.debug("time to matching %s\n\n",time.time()-start)

def ExecuteTask():
    while True:
        request = request_queue.get()
        
        start = time.time()
        task_id = request.task_id
        camera = task_id[:2]  # 'c3/ 1'
        total_img = recover_image(request.width,request.height,request.overlap_data,request.overlap_blocks)
        # save and debug
        out_im = save_offload_imgs(total_img,task_id)

        end_1 = time.time()
        model_index = select_model(request.overlap_blocks,task_id,camera_iou[camera])
        selected_model = model_list[model_index]
        results = selected_model(out_im)
        log.logger.debug("task id %s model index %s time preprocess %s inference %s",task_id,model_index,end_1-start,time.time()-end_1)

        # Results
        results.save('./outputs')  # or .show()
        np_result = results.pandas().xyxy[0].values
        remove_list = []
        for i in range(len(np_result)):
            obj = np_result[i]
            if obj[-1] not in ['car','truck','bus']:
                remove_list.append(i)
        np_result = np.delete(np_result,remove_list,axis=0)  # 删掉traffic light之类的
        task_iou = Task(task_id,None,None,np_result)
        iou_queue.put(task_iou)

        if len(np_result) == 0:
            np_result = config.backup_result
        num_sh = 0
        for object in sharing_object_list:
            obj_info = []
            if camera not in object.camera_list and object.state != 'stop':  
                obj_info.extend(object.box[0])
                obj_info.extend([object.confidence,555,100])
                obj_info = [float(i) for i in obj_info]
                log.logger.debug("obj info %s",obj_info)
                np_result = np.append(np_result,np.array([obj_info]),axis=0)
                num_sh += 1
        log.logger.debug("%s sharing objects",num_sh)
        

        # return results
        end_3 = time.time()
        byte_results = pickle.dumps(np_result)
        PushResult(task_id,byte_results,config.addr[camera])
        task_start = start_time[task_id]
        global inference_time
        inference_time.append(time.time()-task_start)
        log.logger.debug("mean inference %s",np.mean(inference_time))
        
        np_box = pickle.loads(request.sharing_box)  
        task = Task(task_id,out_im,np_box,np_result)
        task_queue.put(task)
        
def analyse_iou():
    while True:
        task_iou = iou_queue.get()
        task_id = task_iou.task_id
        camera = task_id[:2]  # 'c3/ 1'
        np_result = task_iou.result
        
        iou_list = []
        for i in range(len(np_result)):
            for j in range(i+1,len(np_result)):
                iou = compute_iou(np_result[i][:-1],np_result[j][:-1])
                iou_list.append(iou)
        if len(iou_list) > 0:
            m_iou = np.mean(iou_list)
        else:
            m_iou = 0
        global camera_iou
        camera_iou[camera] = m_iou

class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def SendImage(self, request, context):
        print("ip address: ",context.peer())
        # for _ in range(4):
        log.logger.debug("receive %s",request.task_id)
        global start_time
        start_time[request.task_id] = time.time()
        request_queue.put(request)
        return helloworld_pb2.HelloReply(message='Hello!')


def StartServer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=(('grpc.max_send_message_length', int(9291456)),("grpc.max_receive_message_length",int(9291456))))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port(config.addr["server"])
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    log_file = "./log/" + local_role_ + ".log"
    log = Logger(log_file,level='debug')

    os.chdir(sys.path[0])

    image_receiver_t = threading.Thread(target=StartServer,args=(),name="ImageReceiver")
    task_executer_t = threading.Thread(target=ExecuteTask,args=(),name="TaskExecuter")
    task_executer_t1 = threading.Thread(target=ExecuteTask,args=(),name="TaskExecuter1")
    task_executer_t2 = threading.Thread(target=ExecuteTask,args=(),name="TaskExecuter2")
    task_executer_t3 = threading.Thread(target=ExecuteTask,args=(),name="TaskExecuter3")
    object_manager_t = threading.Thread(target=sharing_object_manager,args=(),name="ObjectManager")
    iou_analyser_t = threading.Thread(target=analyse_iou,args=(),name="IOUAnalyser")

    image_receiver_t.start()
    task_executer_t.start()
    task_executer_t1.start()
    task_executer_t2.start()
    task_executer_t3.start()
    object_manager_t.start()
    iou_analyser_t.start()

    image_receiver_t.join()
    task_executer_t.join()
    task_executer_t1.join()
    task_executer_t2.join()
    task_executer_t3.join()
    object_manager_t.join()
    iou_analyser_t.join()