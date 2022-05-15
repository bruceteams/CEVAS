# encoding: utf-8
from __future__ import print_function
from concurrent import futures
import logging

import grpc
from matplotlib import scale
import helloworld_pb2
import helloworld_pb2_grpc
import time,os,sys,pickle
from PIL import Image
from io import BytesIO
import numpy as np
import uuid,threading,queue
from cal_iou import compute_iou,evaluate, object_2_list
from config_our.cluster_config import Logger,cached_object_list,region_list,Task
from tool_client import *
import cv2,argparse


result_queue = queue.Queue()
id_queue = queue.Queue()
current_shifted_results = []
# response latency/iou/data size
metric_dict = {}
bandwidth = 10

def ImageSender():
    os.chdir(sys.path[0])
    time.sleep(10)
    idle = 1
    for num in range(idle,300,idle):
        img_name = str(num) + '.jpg'
        img_path = root_path + img_name
        global start_time
        start_time = time.time()
        log.logger.debug("start process %s",img_name)

        img = Image.open(img_path)
        frame1 = cv2.imread(root_path + str(num-idle) + '.jpg',1)
        frame2 = cv2.imread(img_path,1)

        start = time.time()
        related_blocks = find_related_blocks(region_camera,cached_result)
        flow = obtain_optical_flow(related_blocks,frame1,frame2)

        end_1 = time.time()
        offload_blocks = find_new_objects(flow,local_role_,cached_result)

        end_2 = time.time()
        offload_bytes,offload_blocks,sharing_box,shifted_results = process_cached_results(img,flow,cached_result,local_role_,offload_blocks)

        end_3 = time.time()
        task_id = local_role_ + "/ " + str(num)

        log.logger.debug("%s time consume for preprocess %s %s %s",task_id,end_1-start,end_2-end_1,time.time()-end_2)
        PushImage(img.width,img.height,offload_bytes,offload_blocks,sharing_box,task_id)
        global data_size 
        data_size = sys.getsizeof(offload_bytes+offload_blocks+sharing_box)
        

        global current_shifted_results
        current_shifted_results = shifted_results

        time.sleep(0.3)


def PushImage(im_width,im_height,overlap_data,overlap_blocks,sharing_boxes,task_id):
    try:
        with grpc.insecure_channel(config.addr["server"],options=(('grpc.max_send_message_length', int(9291456)),("grpc.max_receive_message_length",int(9291456)))) as channel:
            stub = helloworld_pb2_grpc.GreeterStub(channel)
            img_request = helloworld_pb2.ImageRequest(width=int(im_width), height=int(im_height), overlap_data=overlap_data,overlap_blocks=overlap_blocks,task_id=task_id,sharing_box=sharing_boxes)
            response = stub.SendImage(img_request)
    except Exception as e:
        time.sleep(0.01)
        print(dir(e))                  
        code_name = e.code().name       
        code_value = e.code().value    
        details = e.details()           
        print(code_name)
        print(code_value)
        # print(json.loads(details))

class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def SendResults(self, request, context):
        print("ip address: ",context.peer())
        np_result = pickle.loads(request.results)
        task_id = request.task_id
        task = Task(task_id,None,None,np_result)
        result_queue.put(task)
        # id_queue.put(task_id)
        return helloworld_pb2.HelloReply(message='Hello!')


def StartServer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=(('grpc.max_send_message_length', int(9291456)),("grpc.max_receive_message_length",int(9291456))))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port(config.addr[local_role_])
    server.start()
    server.wait_for_termination()

def ResultProcess():
    while True:
        global cached_result
        # task_id = id_queue.get()
        task = result_queue.get()
        task_id = task.task_id
        np_result = task.result
        response_latency = time.time()-start_time + data_size/(bandwidth*1024*1024) 
        offload_result = process_array_result(np_result,local_role_)
        current_result = offload_result + current_shifted_results  
        current_list = object_2_list(current_result)
        iou = evaluate(task_id,current_list)

        log.logger.debug("img: %s, time %s iou: %s list %s",task_id,response_latency,iou,current_list)
        if int(iou) >= 0: 
            global metric_dict
            metric_dict.setdefault("latency",[]).append(response_latency)
            metric_dict.setdefault("iou",[]).append(iou)
            metric_dict.setdefault("data",[]).append(data_size/6220833)  
        
            draw_box(task_id,current_list[:,:4],(0, 255, 0)) 

        for obj in current_result:
            if obj.type == 'virtual' or obj.label not in ['car','truck','bus']:
                current_result.remove(obj)
        cached_result = current_result
        log.logger.debug("latency %s iou %s data_size %s\n\n",np.mean(metric_dict["latency"]),np.mean(metric_dict["iou"]),np.mean(metric_dict["data"]))
        
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='camera role')
    parser.add_argument('--role', type=str, help='specify the role of this camera demo, example:c3')
    args = parser.parse_args()

    global local_role_
    local_role_ = args.role
    # local_role_ = 'c1'
    root_path = "./cityflow/" + local_role_ + '/images/'
    cached_result = cached_object_list[local_role_]
    region_camera = region_list[local_role_]

    logging.basicConfig()
    log_file = "./log/" + local_role_ + ".log"
    log = Logger(log_file,level='debug')

    image_sender_t = threading.Thread(target=ImageSender,args=(),name="ImageSender")
    result_receiver_t = threading.Thread(target=StartServer,args=(),name="ResultReceiver")
    result_process_t = threading.Thread(target=ResultProcess,args=(),name="ResultProcess")


    image_sender_t.start()
    result_receiver_t.start()
    result_process_t.start()

    image_sender_t.join()
    result_receiver_t.join()
    result_process_t.join()