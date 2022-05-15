# encoding: utf-8
from pyexpat import features
import threading
from turtle import color, pos
from unittest import result
import numpy as np
import logging
from logging import handlers
import sys,os

addr = {
    "server": 'localhost:70051',
    "c1": 'localhost:90052',
    "c2": 'localhost:90058',
    "c3": 'localhost:80052',
    "c4": 'localhost:80058'
}


img_config = {
    "height": 1080,
    "width": 1920,
    "height-num": 6,
    "height-pixel": 180,
    "width-num": 6,
    "width-pixel": 320
}

def judge_blocksof_box(box):   
    reside_blocks = []
    remain = int(box[0] / img_config['width-pixel']) + 1  # left
    factor = int(box[1] / img_config['height-pixel'])  # top
    top_left = factor * img_config['width-num'] + remain

    remain = int(box[2] / img_config['width-pixel']) + 1   # right
    factor = int(box[3] / img_config['height-pixel'])  # down
    right_down = factor * img_config['width-num'] + remain

    height = int((right_down - top_left) / img_config['width-num'])  
    diff = int((right_down - top_left) % img_config['width-num'])  

    for i in range(height+1):
        for j in range(diff+1):
            b_num = top_left + i * img_config['width-num'] + j  
            reside_blocks.append(b_num)
    
    return reside_blocks

def find_intersection(rec1,rec2):
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    return [left_line,top_line,right_line,bottom_line]

class Task:
    def __init__(self,task_id,data,np_box,result):
        self.task_id = task_id # image id
        self.data = data  # image data
        self.sharing_box = np_box
        self.result = result

class Object:
    def __init__(self,label,confidence,box,o_type,blocks=None):
        self.label = label
        self.confidence = confidence
        self.box = box
        self.blocks = blocks  # the object lies in
        self.type = o_type
    def update_blocks(self):
        correct_box = find_intersection(self.box,[1,1,1919,1079])
        self.box = correct_box
        new_blocks = judge_blocksof_box(self.box) 
        self.blocks = new_blocks

class Sharing_Object:
    def __init__(self,label,confidence,box,camera_list,num,pos=None,color=None,features=None):
        self.label = label
        self.confidence = confidence
        self.box = box   
        self.camera_list = camera_list  
        self.pos = pos   
        self.color = color 
        self.features = features  
        self.state = None
        self.num = num  

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        th = handlers.RotatingFileHandler(filename=filename,mode='w',backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

# 3 types region of 4 cameras
region_c1 = {
    "back": [],
    "income": [],
    "leave": [],
    "overlap": [],
    "leave-edge": [],   
    "neighnor": [],    
    "back-income": []  
}
region_c2 = {
    "back": [],
    "income": [],
    "leave": [],
    "overlap": [],
    "leave-edge": [],  
    "neighnor": [],
    "back-income": []   
}
region_c3 = {
    "back": [],
    "income": [],
    "leave": [],
    "overlap": [],
    "leave-edge": [],   
    "neighnor": [],    
    "back-income": []   
}
region_c4 = {
    "back": [],
    "income": [],
    "leave": [],
    "overlap": [],
    "leave-edge": [],   
    "neighnor": [],
    "back-income": []   
}
region_list = {"c1": region_c1,"c2":region_c2,"c3": region_c3,"c4":region_c4}

# the results of frame 0
object_c1 = [Object('car',0.7,[0.435,525.304,72.047,642.464],'normal',[13,19]), 
             Object('car',0.68,[1077.991,243.738,1127.452,289.515],'normal',[10])]

object_c2 = [Object('car',0.7,[1105.027,162.912,1195.504,228.622],'normal',[4,10]), 
             Object('car',0.68,[1257.253,259.501,1352.031,320.491],'normal',[10,11]), 
             Object('car',0.64,[804.677,123.897,867.303,177.651],'normal',[3]), 
             Object('car',0.62,[988.056,124.775,1069.671,180.933],'normal',[4,10]), 
             Object('car',0.53,[398.572,436.441,880.547,639.031],'overlap',[14,15,20,21]),
             Object('car',0.53,[1201.995,315.523,1414.255,471.26],'normal',[10,11,16,17]),
             Object('car',0.62,[903.617,223.389,999.032,318.991],'normal',[9,10]), 
             Object('car',0.53,[877.792,173.406,950.593,225.526],'normal',[3,9])]

object_c3 = [Object('car',0.7,[474.271,430.879,904.072,665.721],'overlap',[14,15,20,21]), 
             Object('car',0.68,[694.006,127.418,743.576,169.064],'normal',[3]), 
             Object('car',0.64,[662.929,226.673,773.612,333.755],'normal',[9]), 
             Object('car',0.62,[477.762,238.099,597.037,341.474],'normal',[8]), 
             Object('car',0.53,[528.527,200.319,614.089,280.296],'normal',[8])]

object_c4 = [Object('car',0.7,[1309.505,394.025,1672.976,567.2],'overlap',[17,18,23,24]),
             Object('car',0.7,[693.885,293.179,804.973,372.08],'normal',[9,15]), 
             Object('car',0.68,[588.342,250.744,670.084,365.962],'normal',[8,9,14,15]), 
             Object('car',0.62,[672.12,334.749,792.176,454.709],'normal',[9,15]), 
             Object('car',0.53,[416.806,215.791,600.164,468.935],'normal',[8,14])]

cached_object_list = {'c1': object_c1,"c2":object_c2,'c3': object_c3,"c4":object_c4}

backup_result = np.array([[1 ,2 ,3 ,4 ,0.31, 555 , 100]])