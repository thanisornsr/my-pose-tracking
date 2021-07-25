from joint_flow_utils import *
import glob
import numpy as np
# from yolo_classes import get_cls_dict
# from yolo_with_plugins import get_input_shape, TrtYOLO

import cv2
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit

import scipy 
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from skimage.transform import resize
from timeit import default_timer as timer

#live config
is_draw_skeleton = True
is_show = True
is_save = False

# config 
# myBatchSize = 1
BATCH_SIZE = 1
# myInputShape = (368,368)
input_shape = (368,368)
# myOutputShape = (46,46)
output_shape = (46,46)
n_keypoints = 15
n_limbs = 16
USE_FP16 = False # USE FP32

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
print('Connecting Camera: Done')

# build openpose
print('Generating Openpose: ...')

dummy_input_batch = np.zeros((BATCH_SIZE,*input_shape,3),dtype = np.float32)

target_dtype = np.float16 if USE_FP16 else np.float32

f2 = open("open_pose_engine.trt","rb")
runtime2 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine2 = runtime2.deserialize_cuda_engine(f2.read())
context2 = engine2.create_execution_context()
output2 = np.empty([BATCH_SIZE,*output_shape,175], dtype = target_dtype)
d_input2 = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
d_output2 = cuda.mem_alloc(1 * output2.nbytes)

bindings2 = [int(d_input2),int(d_output2)]
stream2 = cuda.Stream()

print('Generating Openpose: Done')

# build TFF # To modify
print('Generating TFF: ...')

dummy_input_batch0 = np.zeros((BATCH_SIZE,*input_shape,6),dtype = np.float32)

# target_dtype = np.float16 if USE_FP16 else np.float32

f3 = open("TFF_engine.trt","rb")
runtime3 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine3 = runtime3.deserialize_cuda_engine(f3.read())
context3 = engine3.create_execution_context()
output3 = np.empty([BATCH_SIZE,*output_shape,124], dtype = target_dtype)

d_input3 = cuda.mem_alloc(1 * dummy_input_batch0.nbytes)
d_output3 = cuda.mem_alloc(1 * output3.nbytes)

bindings3 = [int(d_input3),int(d_output3)]
stream3 = cuda.Stream()

print('Generating TFF: Done')

print('Warming up ...') # To modify
prediction0 = predict_open(dummy_input_batch,context2,bindings2,d_input2,d_output2,stream2,output2)
prediction = predict_TFF(dummy_input_batch0,context3,bindings3,d_input3,d_output3,stream3,output3)
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
Q = {}

frame_id = 0
pose_id = 0

while True:
	start = timer()
	ret, img = cap.read() 

	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = img / 255.0
	img = np.reshape(resize(img,input_shape),(1,*input_shape,-1))
	img = img.astype(np.float32)

	if frame_id == 0:
		poses,pose_id = joint_flow_pipeline_first_frame(pose_id,img,context2,bindings2,d_input2,d_output2,stream2,output2)
		poses = set_valid_to_poses(poses)
		poses = convert_kps_to_global(poses,img_w,img_h)

		Q[frame_id] = poses
		poses0 = poses
		img0 = img 
		end = timer()
		frame_id = frame_id + 1
	else:
		poses,pose_id = joint_flow_pipeline(pose_id,img0,img,poses0,context3,bindings3,d_input3,d_output3,stream3,output3)
		poses = set_valid_to_poses(poses)
		poses = convert_kps_to_global(poses,img_w,img_h)

		Q[frame_id] = poses
		poses0 = poses
		img0 = img 
		end = timer()
		frame_id = frame_id + 1

	img_out = np.copy(img)

	if is_draw_skeleton:
		Qs = Q[frame_count]

		temp_FPS = 1/(end-start)
		to_add_str = 'FPS: {:.2f}'.format(temp_FPS)
		c_code = (255,0,0)
		pos_FPS = (i_w-200,i_h-100)
		cv2.putText(img_out,to_add_str,pos_FPS,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)

		for qs in Qs:
			tid = qs.id
			tkps = qs.global_joints
			tvs = qs.valids
			# tbbox = [xmax,ymax,xmin,ymin]
			x_list = [tkps[x][1] for x in list(tkps.keys())]
			y_list = [tkps[x][0] for x in list(tkps.keys())]

			xmax = np.max(x_list)
			xmin = np.min(x_list)
			ymax = np.max(y_list)
			ymin = np.min(y_list)
			tbbox = [xmin,ymin,xmax-xmin,ymax-ymin]
			# print(tbbox)

			#put id in image
			pos_txt = (int(xmax-10),int(ymax-50))
			# print(pos_txt)
			# pos_txt = (900,360)
			cv2.putText(img_out,str(qs.id),pos_txt,cv2.FONT_HERSHEY_COMPLEX,2,c_code,thickness=2)

			#draw skeleton

			# line
			ci = 0
			for sk in skeleton_list:
				p1,p2 = sk
				if tvs[p1] == 1 and tvs[p2] == 1:
					y1,x1 = tkps[p1]
					y2,x2 = tkps[p2]
					tc = color_list[ci%6]
					cv2.line(img_out,(x1,y1),(x2,y2),tc,2)
				ci = ci + 1
			# dot
			for i in range(len(tvs)):
				if tvs[i] == 1:
					y,x = tkps[i]
					cv2.circle(img_out,(x,y),2,(0,0,255),3)
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
