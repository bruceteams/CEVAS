# encoding: utf-8
from copyreg import pickle
from doctest import OutputChecker
from pickletools import uint8
import threading
from turtle import right
import numpy as np
import logging,time
from logging import handlers
from PIL import Image
import cv2,sys,math,os
import config_our.cluster_config as config
import copy,pickle
from config_our.tool import MyThread

move_th = 0.1  
move_th_share = 0.05
pixel_num_new = 1000
pixel_num_change = 8000

log_file = "./log/tool_client.log"
log = config.Logger(log_file,level='debug')

def cal_optical_flow(frame1,frame2):
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def find_intersection(rec1,rec2):
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    return [left_line,top_line,right_line,bottom_line]

def block_to_box(block):
    block = int(block) - 1  

    width_interval = config.img_config['width-pixel']
    height_interval = config.img_config['height-pixel']
    left = int((block % 6)) * width_interval
    top = int((block / 6)) * height_interval

    return [left,top,left+width_interval,top+height_interval]

def add_area(box,x_offset,y_offset):
    if box[1] % config.img_config["height-pixel"] < 1 and y_offset < 0:
        box[1] -= 2
    if box[3] % config.img_config["height-pixel"] > 179 and y_offset > 0:
        box[3] += 2
    if box[0] % config.img_config["width-pixel"] < 1 and x_offset < 0:
        box[0] -= 2
    if box[2] % config.img_config["width-pixel"] > 319 and x_offset > 0:
        box[2] += 2
    return box

def shift_box(box,flow):
    int_box = [int(float(i)) for i in box]
    x_offset = np.mean(flow[int_box[1]:int_box[3],int_box[0]:int_box[2],0])
    y_offset = np.mean(flow[int_box[1]:int_box[3],int_box[0]:int_box[2],1])
    offset = math.sqrt(x_offset ** 2 + y_offset ** 2)

    left = float(box[0]) + x_offset 
    right = float(box[2]) + x_offset 
    top = float(box[1]) + y_offset 
    down = float(box[3]) + y_offset 

    box = [left,top,right,down]

    box = add_area(box,x_offset,y_offset)

    new_box = find_intersection(box,[1,1,config.img_config["width"]-1,config.img_config["height"]-1])

    return new_box,offset

def collect_offload_data(img_array,blocks):
    total_data = []
    for block in blocks:
        box = block_to_box(block)
        block_data = img_array[box[1]:box[3], box[0]:box[2], :]
        total_data.append(block_data)
    total_data = np.array(total_data)
    return total_data

def count_no_zero(box,matrix):
    box = [int(i) for i in box]
    box_pixel = matrix[box[1]:box[3], box[0]:box[2]]
    no_zero = np.nonzero(box_pixel)  # row /column
    total_num = len(no_zero[0])

    return total_num


def find_new_objects(flow,camera_role,cached_results):
    offload_blocks = []
    candidate_list = []  # 备选block
    back_income = config.region_list[camera_role]["back-income"]
    leave_edge = config.region_list[camera_role]["leave-edge"]
    leave_neighnor = config.region_list[camera_role]["neighnor"]
    candidate_list.extend(back_income)
    candidate_list.extend(leave_edge)

    t_flow = flow[:,:,0] ** 2 + flow[:,:,1] ** 2  # total offset
    t_flow = np.int_(t_flow)

    for block in candidate_list:
        block_box = block_to_box(block)
        total_num = count_no_zero(block_box,t_flow)  # 该block内所有光流值不为0的pixel的数量
        existing_num = 0
        for object in cached_results:
            object_blocks = object.blocks
            if block in object_blocks:
                old_box = object.box
                new_box = find_intersection(block_box,old_box)
                temp_num = count_no_zero(new_box,t_flow)
                existing_num += temp_num
        remain_num = total_num - existing_num
        if block in back_income and remain_num > pixel_num_new:  # 代表有新物体产生
            offload_blocks.append(block)
        elif block in leave_edge and remain_num > pixel_num_change: # 有物体从overlapping区域离开
            # offload_blocks.append(block)
            offload_blocks.extend(leave_edge)
            offload_blocks.extend(leave_neighnor)   # 为了完整的检测该物体
    
    return offload_blocks    


def obtain_optical_flow(related_blocks,frame1,frame2):
    start = time.time()
    optical_flow = np.zeros((config.img_config['height'],config.img_config['width'],2)) 
    box_list = []  
    thread_list = []
    for block in related_blocks:
        box = block_to_box(block) 
        box_list.append(box)
        num = box_list.index(box)
        # log.logger.debug("box %s block %s",box,block)
        crop_1 = frame1[box[1]:box[3], box[0]:box[2]]
        crop_2 = frame2[box[1]:box[3], box[0]:box[2]]
        # thread_1 = MyThread(cal_optical_flow, args=(crop_1, crop_2))
        # thread_1.start()
        thread_list.append(MyThread(cal_optical_flow, args=(crop_1, crop_2)))
        thread_list[num].start()
        # flow_block = cal_optical_flow(crop_1,crop_2)
    for t in thread_list:
        t.join()
    
    for t in thread_list:
        flow_block = t.get_result()
        # log.logger.debug("%s",flow_block)
        index = thread_list.index(t)
        box = box_list[index]
        optical_flow[box[1]:box[3], box[0]:box[2], :] = flow_block  # 一个block的光流值
    print(time.time()-start)
    return optical_flow


def find_related_blocks(region_camera,cached_results):
    related_blocks = []
    for object in cached_results:
        if object.type != "virtual":  # virtual sharing objects
            related_blocks.extend(object.blocks)

    related_blocks.extend(region_camera['leave-edge'])
    related_blocks.extend(region_camera['back-income'])
    related_blocks = list(set(related_blocks))  # 去重

    return related_blocks

def inter(a,b):
    return list(set(a)&set(b))

def is_sharing_object(blocks,camera_role):
    config_region = config.region_list[camera_role]
    overlap_region = config_region['overlap']

    l1 = inter(overlap_region,blocks)  # 和overlapping区域的交集

    if l1:
        return True
    else:
        return False


def process_cached_results(img,flow,cached_result,camera_role,offload_blocks):
    shifted_results = []   
    sharing_objects_box = []   
    img_array = np.array(img)

    for object in cached_result:

        is_sharing = is_sharing_object(object.blocks,camera_role)

        shifted_box,offset = shift_box(object.box,flow)
        new_object = copy.deepcopy(object)
        new_object.box = shifted_box
        new_object.update_blocks()   

        blocks = new_object.blocks
        if not (set(blocks) < set(offload_blocks)):  
            object_offload = False
            if (offset > move_th) or (offset > move_th_share and new_object.type == "overlap"): 
                offload_blocks.extend(blocks)
                object_offload = True
            
            is_sharing_2 = is_sharing_object(blocks,camera_role)
            if (is_sharing is False) and (is_sharing_2 is True):
                offload_blocks.extend(blocks)
                object_offload = True

            if not object_offload:   
                shifted_results.append(new_object)
                if new_object.type == "overlap":
                    info = []
                    info.extend(new_object.box)
                    info.extend([new_object.confidence,-1,-1])  
                    sharing_objects_box.append(info)

    offload_blocks = list(set(offload_blocks)) 
    offload_blocks.sort()     
    offload_data = collect_offload_data(img_array,offload_blocks)

    offload_blocks = np.array(offload_blocks)
    offload_blocks = offload_blocks.tobytes()
    offload_bytes = offload_data.tobytes()  # (n,320,180,3)


    sharing_objects_box = np.array(sharing_objects_box,dtype=int)
    sharing_box = pickle.dumps(sharing_objects_box)

    return offload_bytes,offload_blocks,sharing_box,shifted_results

def draw_grid(img, line_color=(0, 0, 0), thickness=2, type_=cv2.LINE_AA, pxstep=50,pystep=50):
    x = pxstep
    y = pystep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pystep
    return img

def draw_box(task_id,boxes,color):
    paths = task_id.split()  # 'c3/ 10'
    root_path = "./cityflow/" + paths[0] + 'images/'
    img_path = root_path + paths[1] + '.jpg'
    output_path = './outputs/client/' + paths[0] + paths[1] + '.jpg'
    if os.path.exists(output_path):
        img = cv2.imread(output_path,1)
    else:
        img = cv2.imread(img_path,1)
    for box in boxes:
        box = [int(float(i)) for i in box]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)  
    cv2.imwrite(output_path,img)

def process_array_result(offload_result,camera_role):
    offload_result_list = []
    for object_info in offload_result:
        label = object_info[-1]
        confidence = round(object_info[-3],2)
        box = list(object_info[:4])
        if int(object_info[-2]) == 555 or (label not in ['car','truck','bus']):
            type = 'virtual'
        else:
            type = 'normal'
        new_object = config.Object(label,confidence,box,type)
        new_object.update_blocks()

        config_region = config.region_list[camera_role]
        overlap_region = config_region['overlap']

        if new_object.type == 'normal':
            l1 = inter(overlap_region,new_object.blocks)
            if l1:
                new_object.type = 'overlap'
        offload_result_list.append(new_object)
    return offload_result_list


def main():
    root_path = './test/cityflow/demo_images/'
    img_list = ['c1-32.jpg','c2-32.jpg','c3-32.jpg','c4-32.jpg']
    for path in img_list:
        img_path = root_path + path
        img = cv2.imread(img_path,1)
        print(img.shape)
        img_new = draw_grid(img,pxstep=270)
        path_new = 'new-' + path
        img_new_path = root_path + path_new
        cv2.imwrite(img_new_path,img_new)