#!/usr/bin/env python
# encoding: utf-8
from cProfile import label
from config_our.cluster_config import Logger
from tool_client import draw_box
import numpy as np
import config_our.cluster_config as config

log_file = "./log/iou.log"
log = config.Logger(log_file,level='debug')

def read_truth(task_id):
    paths = task_id.split()  # 'c3/ 10'
    root_path = "./cityflow/"
    truth_path = root_path + paths[0] + 'truth.txt'  # ./cityflow/c3/
    labels = []
    with open(truth_path) as f:
        for line in f.readlines():
            result = line.split(',')
            if result[0] == paths[1]:
                labels.append(result[1:])  # from 1
            if int(result[0]) > int(paths[1]):
                break
    return labels

def compute_iou(rec1, rec2):
    rec1 = [float(i) for i in rec1]
    rec2 = [float(i) for i in rec2]
    # computing area of each rectangles
    S_rec1 = (rec1[3] - rec1[1]) * (rec1[2] - rec1[0])
    S_rec2 = (rec2[3] - rec2[1]) * (rec2[2] - rec2[0])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0

def filter_label(labels):
    new_labels = []
    pre_obj_id = -1
    for item in labels:
        if int(item[0]) == -1:
            new_labels.append(item) 
        else:
            if int(item[0]) != pre_obj_id:
                pre_obj_id = int(item[0])
                new_labels.append(item)
    return new_labels

def evaluate(task_id,pred):
    labels = read_truth(task_id)
    if len(labels) == 0:
        return -1 
    local_labels = filter_label(labels)
    cla_iou_list = []   
    draw_box(task_id,np.array(local_labels)[:,1:5],(0, 0, 255))

    iou_obj_dict = {}   
    for label in labels:  
        obj_id = label[0]  
        iou_list = []
        for pred_item in pred:
            iou_t = compute_iou(label[1:5],pred_item[:4])
            iou_list.append(iou_t)
        if len(iou_list) > 0:
            iou = max(iou_list)
        else:
            iou = 0
        iou_obj_dict.setdefault(obj_id,[]).append(iou)
    
    for iou_list in iou_obj_dict.values():
        cla_iou_list.append(max(iou_list))
    log.logger.debug("%s %s",cla_iou_list,sum(cla_iou_list)/len(cla_iou_list))
    return sum(cla_iou_list)/len(cla_iou_list)

def object_2_list(object_list):
    result_list = []
    for object in object_list:
        item_list = []
        item_list.extend(object.box)
        item_list.append(object.confidence)
        item_list.append(object.label)
        result_list.append(item_list)
    return np.array(result_list)
 
if __name__=='__main__':
    rect1 = (1572.0, 342.0, 1920.0, 505.0)
    # (top, left, bottom, right)
    rect2 = (1574.896,196.613,1917.851,495.523)
    iou = compute_iou(rect1, rect2)
    print(iou)