
from light_track_utils import *
import glob
import numpy as np
from yolo_classes import get_cls_dict
from yolo_with_plugins import get_input_shape, TrtYOLO
import glob
import os
import cv2
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

from skimage.transform import resize
from timeit import default_timer as timer 

#live config
is_draw_skeleton = True
is_show = False
is_save = True

# config 
h = 416
w = 416
conf_th = 0.95
yolo_model = 'yolov4-416'
BATCH_SIZE = 1
USE_FP16 = False # USE FP32
input_shape =(384, 288)
output_shape = (96,72)
KEY_FRAME = 5


skeleton_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

print('Connecting Camera: ...')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640.0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360.0)
cap.set(cv2.CAP_PROP_FPS,30.0)

if not (cap.isOpened()):
    print("Could not open video device")

ret, frame = cap.read()  
i_h, i_w,_ = frame.shape
img_h = i_h 
img_w = i_w 
print('Connecting Camera: Done')

# build human detector
print('Generating human detector: ...')

hdetector = TrtYOLO(yolo_model, (h, w))

print('Generating human detector: Done')

# build model
print('Generating pose estimator: ...')

dummy_input_batch = np.zeros((BATCH_SIZE,*input_shape,3),dtype = np.float32)

target_dtype = np.float16 if USE_FP16 else np.float32
f2 = open("lt_resnet152_engine.trt","rb")
# f2 = open("lt_resnet_engine.trt","rb")
runtime2 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine2 = runtime2.deserialize_cuda_engine(f2.read())
context2 = engine2.create_execution_context()
output2 = np.empty([BATCH_SIZE,*output_shape,15], dtype = target_dtype)
d_input2 = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
d_output2 = cuda.mem_alloc(1 * output2.nbytes)
bindings2 = [int(d_input2),int(d_output2)]
stream2 = cuda.Stream()

print('Generating pose estimator: Done')

print('Warming up ...')

predict(dummy_input_batch,context2,bindings2,d_input2,d_output2,stream2,output2)
boxes, confs, clss = hdetector.detect(frame,conf_th)

print('Done Warming up!')

if is_save:
    print('Cleaning live imgs folder: ...')
    for filename in sorted(glob.glob('./live_imgs/*.jpg')):
        os.remove(filename)
    print('Cleaning live imgs folder: Done')

# counting down
countdown_sec = 5
while countdown_sec > 0:
    print('Flow track live start in: {}'.format(countdown_sec))
    time.sleep(1)
    countdown_sec = countdown_sec - 1

# Go live!
Q = []
frame_id = 0
pose_id = 0
key_frame = KEY_FRAME
while True:
    start = timer()
    ret, img = cap.read() 
    

    # img_h,img_w,_ = img.shape
    # print('{},{}'.format(img_h,img_w))

    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_RGB = img_RGB / 255.0

    if (frame_id % key_frame) == 0:
        batch_imgs,batch_bboxes,n_human = human_detect_module(hdetector,img,img_RGB,input_shape)
    else:
        last_Q = Q[-1]
        # still_tracking_ID = [tracked_ID[x] for x in range(len(tracked_ID)) if tracked_status[x] == 1]
        still_tracking_ID = [tQ.id for tQ in last_Q]
        bboxes_from_last_frames = [tQ.bbox for tQ in last_Q]
        # batch_bboxes = expand_bboxes(bboxes_from_last_frames,img_h,img_w,20)
        batch_bboxes = bboxes_from_last_frames
        batch_imgs,n_human, no_img = crop_img_from_bboxes(img_RGB,bboxes_from_last_frames,input_shape)
        for idx in reversed(no_img):
            batch_bboxes.pop(idx)

    p_heatmaps = []
    for bimg in batch_imgs:
        img_to_predict = np.reshape(bimg,(1,*input_shape,3))
        tp_heatmaps = predict(img_to_predict,context2,bindings2,d_input2,d_output2,stream2,output2)
        p_heatmaps.append(np.copy(tp_heatmaps[0,:,:,:]))

    p_heatmaps = np.array(p_heatmaps)

    temp_Ji, temp_Vi = heatmaps_to_joints(p_heatmaps)
    # ------------------------------------------------------------------------
    
    new_bboxes,no_joint,temp_Ji = get_bbox_from_joints(batch_bboxes,temp_Ji,temp_Vi,output_shape,img_w,img_h,20)
    for idx in reversed(no_joint):
        # print(len(batch_bboxes))
        # print(idx)
        batch_bboxes.pop(idx)
        temp_Ji.pop(idx)
        temp_Vi.pop(idx)
    n_human = len(temp_Ji)

    if frame_id == 0:
        Q_to_add = []
        for j in range(n_human):
            temp_pose = new_Pose(frame_id,pose_id,temp_Ji[j],batch_bboxes[j],temp_Vi[j])
            Q_to_add.append(temp_pose)
            pose_id = pose_id + 1
        frame_id = frame_id + 1
        current_Q = Q_to_add
        Q.append(Q_to_add)
    else:
        t_score_matrix = cal_scoring_matrix(Q,new_bboxes) 
        pose_id, temp_ids = get_id_to_assign(pose_id,Q,temp_Ji,t_score_matrix,frame_id)
        Q_to_add = []
        for j in range(len(temp_ids)):
            temp_pose = new_Pose(frame_id,temp_ids[j],temp_Ji[j],new_bboxes[j],temp_Vi[j])
            Q_to_add.append(temp_pose)
        frame_id = frame_id + 1
        current_Q = Q_to_add
        Q.append(Q_to_add)
    end = timer()




    img_out = np.copy(img)

    if is_draw_skeleton:

        temp_FPS = 1/(end-start)
        to_add_str = 'FPS: {:.2f}'.format(temp_FPS)
        pos_FPS = (i_w - 200, i_h - 100)
        c_code = (255,0,0)
        cv2.putText(img_out,to_add_str,pos_FPS,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
        for qs in current_Q:
            tid = qs.id
            tbbox = qs.bbox
            tkps = qs.joints
            tvs = qs.valids
            g_kps = np.transpose(tkps) #becuz light track joints is global already
            
            # draw bbox
            c_code = (0,255,0)
            cv2.rectangle(img_out,(int(tbbox[0]),int(tbbox[1])),(int(tbbox[0]+tbbox[2]),int(tbbox[1]+tbbox[3])),c_code,3)
            
            # put id in image
            pos_txt = (int(tbbox[0]+tbbox[2]-50),int(tbbox[1]+tbbox[3]-10))
            cv2.putText(img_out,str(qs.id),pos_txt,cv2.FONT_HERSHEY_COMPLEX,2,c_code,thickness=3)
            
            # line
            ci = 0
            for sk in skeleton_list:
                p1,p2 = sk
                x1,y1 = g_kps[p1,:].tolist()
                x2,y2 = g_kps[p2,:].tolist()
                tc = color_list[ci%6]
                ci = ci + 1
                if tvs[p1] == 1 and tvs[p2] == 1:
                    cv2.line(img_out,(x1,y1),(x2,y2),tc,3)
                    
            # dot
            for i in range(15):
                x,y = g_kps[i,:].tolist()
                if tvs[i] == 1:
                    cv2.circle(img_out,(x,y),2,(0,0,255),4)
                    
    if is_show:
        cv2.imshow('frame', img_out)

    if is_save:
        num_name = '{:07d}'.format(frame_id-1)
        cv2.imwrite('./live_imgs/'+num_name+'.jpg',img_out)

    if not is_show:
        print(frame_id)
        if frame_id == 120:
            break
    
    #Waits for a user input to quit the application    
    if cv2.waitKey(1) & 0xFF == ord('q'):    
        break

cap.release()
cv2.destroyAllWindows()