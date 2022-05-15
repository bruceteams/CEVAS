# encoding: utf-8
import threading
import numpy as np
import logging,time
from logging import handlers, root
import skimage.measure
from PIL import Image
import cv2,sys,math,os,torch
import config_our.cluster_config as config
from tool_client import block_to_box
import torch_model
import torchvision.transforms as T
import grpc,os,sys
import helloworld_pb2
import helloworld_pb2_grpc

resize_img = (128,128)   
color_thres = 9
feature_thres = 0.025
frame_thre = 10
iou_thre = 0.07

# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
encoder = torch_model.ConvEncoder()

encoder.load_state_dict(torch.load('./deep_encoder.pt', map_location=device))
encoder.eval()
encoder.to(device)

log_file = "./log/tool_server.log"
log = config.Logger(log_file,level='debug')

def concat_images(img_list,Col,Row):

    target = Image.new('RGB', (config.img_config["width"] * Col, config.img_config["height"] * Row))
    for row in range(Row):
        for col in range(Col):
            target.paste(img_list[Col*row+col], (0 + config.img_config["width"]*col, 0 + config.img_config["height"]*row))
    
    return target

def select_model(offload_blocks,task_id,m_iou):
    log.logger.debug("camera %s, m_iou %s",task_id,m_iou)
    camera = task_id[:2]  # 'c3/ 87'
    num = int(task_id[4:])
    back_income = config.region_list[camera]["back-income"]
    overlap_list = list(np.frombuffer(offload_blocks, dtype=int))
    ret = list(set(back_income).intersection(set(overlap_list)))
    if num % frame_thre == 0:
        return 2
    elif len(ret) > 0:
        return 1
    else:
        if m_iou == 0:
            return 0
        elif m_iou > iou_thre:
            return 2
        else:
            return 1

def reshape_data(data,in_width,in_height,scale):
    reshaped_np = np.frombuffer(data, dtype=np.uint8)
    scaled_height = int(in_height/scale)
    scaled_width = int(in_width/scale)
    try:
        if reshaped_np.shape[0] == scaled_width * scaled_height * 3:
            print("Validated the size of the incoming byte array!")
    except:
        return
    image_array = np.zeros((in_height, in_width, 3), dtype=np.uint8)
    temp_array = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)

    for i in range(3):
        temp_array[:, :, i] = reshaped_np[i::3].reshape((scaled_height, scaled_width))

    for i in range(scaled_height):  # 96
        for j in range(scaled_width):  # 120
            root_x = i*scale
            root_y = j*scale
            for x in range(root_x,root_x+scale):
                for y in range(root_y,root_y+scale):
                    image_array[x,y,:] = temp_array[i,j,:]
    return image_array

def insert_overlap_blocks(img_array,overlap_blocks,overlap_data):
    block_height = config.img_config['height-pixel']
    block_width = config.img_config['width-pixel']
    overlap_list = np.frombuffer(overlap_blocks, dtype=int)
    # print(overlap_list.shape)
    num = len(overlap_list)
    original_data = np.frombuffer(overlap_data, dtype=np.uint8)
    reshape_array = np.zeros((num,block_height,block_width,3), dtype=np.uint8)
    for i in range(3):
        reshape_array[:,:,:,i] = original_data[i::3].reshape((num, block_height,block_width))
    for b_num in overlap_list:
        box = block_to_box(b_num)
        count = np.argwhere(overlap_list == b_num)
        # print(box)
        img_array[box[1]:box[3], box[0]:box[2],:] = reshape_array[count,:,:,:]
    return img_array

def recover_image(im_width,im_height,overlap_data,overlap_blocks):
    re_pool_img = np.ones((im_height, im_width, 3), dtype=np.uint8)
    total_img = insert_overlap_blocks(re_pool_img,overlap_blocks,overlap_data)
    return total_img

def save_offload_imgs(total_img,task_id):
    root_path = "./outputs/server/"
    paths = task_id.split()
    root_path = root_path + paths[0]
    img_path = root_path + paths[1] + '.jpg'
    out_im = Image.fromarray(total_img)
    out_im.save(img_path)
    return out_im

def extract_color(img,box):
    diff_x = (box[2] - box[0]) * 0.25
    diff_y = (box[3] - box[1]) * 0.25
    box = [box[0]+diff_x,box[1]+diff_y,box[2]-diff_x,box[3]-diff_y]
    object = img.crop(box)
    result = object.convert('P', palette=Image.ADAPTIVE, colors=1)  # 1 main colors
    result = result.convert('RGB')
    main_colors = result.getcolors()

    col_extract = []
    for count, col in main_colors:
        col_extract.append([col[i] for i in range(3)])
    return np.array(col_extract)

def load_image_tensor(img):
    image_tensor = T.ToTensor()(img)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    return image_tensor

def extract_feature(img,box):
    cropped = img.crop(box)  # (left, upper, right, lower)
    out = cropped.resize(resize_img,Image.BILINEAR)
    image_tensor = load_image_tensor(out)

    with torch.no_grad():
        feature_list = encoder(image_tensor)
    flattened_f_list = []
    for f in feature_list:
        f = f.cpu().detach().numpy()
        flattened_f = f.reshape((1, -1))
        flattened_f_list.append(flattened_f)

    return flattened_f_list

def cal_weighted_E_dis(x,y):
    from scipy.spatial.distance import pdist,cdist
    X = np.vstack([x,y])
    d1 = pdist(X,'euclidean')[0]
    d1 = d1/float(x.size)
    
    return d1

def compare_two_feature_list(feature_list1,feature_list2):
    num = len(feature_list1)
    dist_list = []
    for i in range(num):
        f_1 = feature_list1[i]
        f_2 = feature_list2[i]
        f_dist = cal_weighted_E_dis(f_1,f_2) 
        dist_list.append(f_dist)
    dist = sum(dist_list)
    return dist

def merge_feature(f1,f2):
    new_f = []
    for i in range(len(f1)):
        temp = (f1[i] + f2[i])/2
        new_f.append(temp)
    return new_f

def compare_existing_object(object,color,features):
    color_pre = np.array([object.color])     
    feature_pre = object.features
    color_dis = compare_two_feature_list(color_pre,color)
    feature_dis = compare_two_feature_list(feature_pre,features)
    log.logger.debug("color dis %s, feature dis %s\n\n",color_dis,feature_dis)
    if color_dis < color_thres and feature_dis < feature_thres:
        # object.color = (color_pre + color)[0]/2
        object.color = color[0]
        object.features = merge_feature(feature_pre,features)
        return True
    else:
        return False    

def PushResult(task_id,byte_result,addr):
    try:
        with grpc.insecure_channel(addr,options=(('grpc.max_send_message_length', int(9291456)),("grpc.max_receive_message_length",int(9291456)))) as channel:
            stub = helloworld_pb2_grpc.GreeterStub(channel)
            print(type(byte_result),type(task_id))
            result_request = helloworld_pb2.ResultRequest(task_id=task_id,results=byte_result)
            response = stub.SendResults(result_request)
            print(response.message)
    except Exception as e:
            time.sleep(0.01)
            print(dir(e))                   
            code_name = e.code().name       
            code_value = e.code().value     
            details = e.details()           
            print(code_name)
            print(code_value)
            # print(json.loads(details))