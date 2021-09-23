from own_track_utils import *
import glob
import numpy as np
from yolo_classes import get_cls_dict
from yolo_with_plugins import get_input_shape, TrtYOLO

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
input_shape = (256,192)
output_shape = (64,48)
FS_input_shape = (64,64)

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

# build human detector
print('Generating human detector: ...')

trt_yolo = TrtYOLO(yolo_model, (h, w))

print('Generating human detector: Done')

# build model
print('Generating pose estimator: ...')

dummy_input_batch = np.zeros((BATCH_SIZE,*input_shape,3),dtype = np.float32)

target_dtype = np.float16 if USE_FP16 else np.float32
f2 = open("resnet152_engine.trt","rb")
# f2 = open("resnet_engine.trt","rb")
runtime2 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine2 = runtime2.deserialize_cuda_engine(f2.read())
context2 = engine2.create_execution_context()
output2 = np.empty([BATCH_SIZE,64,48,15], dtype = target_dtype)
d_input2 = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
d_output2 = cuda.mem_alloc(1 * output2.nbytes)
bindings2 = [int(d_input2),int(d_output2)]
stream2 = cuda.Stream()

print('Generating pose estimator: Done')

print('Generating face scorer: ...')

dummy_input_batch3 = np.zeros((BATCH_SIZE,64,64,2),dtype = np.float32)

target_dtype = np.float16 if USE_FP16 else np.float32
f3 = open("FS_engine.trt","rb")
runtime3 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine3 = runtime3.deserialize_cuda_engine(f3.read())
context3 = engine3.create_execution_context()
output3 = np.empty([BATCH_SIZE,1], dtype = target_dtype)
d_input3 = cuda.mem_alloc(1 * dummy_input_batch3.nbytes)
d_output3 = cuda.mem_alloc(1 * output3.nbytes)
bindings3 = [int(d_input3),int(d_output3)]
stream3 = cuda.Stream()

print('Generating face scorer: Done')

print('Warming up ...')

img = frame
prediction = predict(dummy_input_batch,context2,bindings2,d_input2,d_output2,stream2,output2)
score = predictFS(dummy_input_batch3,context3,bindings3,d_input3,d_output3,stream3,output3)
boxes, confs, clss = trt_yolo.detect(img,conf_th)
# print(score)
print('Done Warming up!')

if is_save:
    print('Cleaning live imgs folder: ...')
    for filename in sorted(glob.glob('./live_imgs/*.jpg')):
        os.remove(filename)
    print('Cleaning live imgs folder: Done')

# counting down
countdown_sec = 5
while countdown_sec > 0:
    print('Own track live start in: {}'.format(countdown_sec))
    time.sleep(1)
    countdown_sec = countdown_sec - 1

# Go live!
Q = []
frame_id = 0
pose_id = 0
total_FPS = 0
while True:
	start = timer()
	ret, img_ori = cap.read()

	temp_bboxes = img_to_bboxes(img_ori,trt_yolo,i_h,i_w)
	img = cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB)
	img = img / 255.0

	batch_imgs = []
	for i_bbox in temp_bboxes:
		o_crop = img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
		if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
			print('detect empty image')
			continue

		o_crop = resize(o_crop,input_shape)
		o_crop = o_crop.astype('float32')
		batch_imgs.append(o_crop)

	nh = len(batch_imgs)
	# batch_imgs = np.array(batch_imgs)
	# batch_imgs = tf.convert_to_tensor(batch_imgs,dtype=tf.float32)
	p_heatmaps = []
	
	for bimg in batch_imgs:
		img_to_predict = np.reshape(bimg,(1,*input_shape,3))
		tp_heatmaps = predict(img_to_predict,context2,bindings2,d_input2,d_output2,stream2,output2)
		p_heatmaps.append(np.copy(tp_heatmaps[0,:,:,:]))

	p_heatmaps = np.array(p_heatmaps)

	temp_Ji,temp_Vi = heatmaps_to_joints(p_heatmaps)
	joints_input_size = [x * 4.0 for x in temp_Ji]

	if frame_id ==0:
		Q_to_add = []
		pfaces = []
		for j in range(len(temp_Ji)):
			temp_bbox = temp_bboxes[j]
			temp_j = temp_Ji[j] #local
			temp_v = temp_Vi[j]
			temp_gj = get_global_joints(temp_bbox,temp_j)
			#get face

			fimg = batch_imgs[j]
			# print(fimg.shape)
			temp_joints = joints_input_size[j]
			tjoints = np.transpose(temp_joints.copy())
			gimg = rgb2gray(fimg)
			gimg = gimg[...,np.newaxis]
			f = get_face_from_joints(gimg,tjoints)
			f = resize(f,(64,64,1))
			pfaces.append(f)

			temp_pose = new_Pose(frame_id,pose_id,temp_j,temp_gj,temp_bbox,temp_v)
			Q_to_add.append(temp_pose)
			pose_id = pose_id + 1
		frame_id = frame_id + 1
		Q.append(Q_to_add)
		# pfaces = [x.face for x in Q[-1]]
		pid = [x.id for x in Q[-1]]
	else:
		#get faces
		faces = []
		for j in range(len(temp_Ji)):
			fimg = batch_imgs[j]
			temp_joints = joints_input_size[j]
			tjoints = np.transpose(temp_joints.copy())
			gimg = rgb2gray(fimg)
			gimg = gimg[...,np.newaxis]
			f = get_face_from_joints(gimg,tjoints)
			f = resize(f,(64,64,1))
			faces.append(f)


		mat_score = np.zeros((len(pid),len(faces)))
		# print(mat_score)
		for i in range(len(pid)):
			for j in range(len(faces)):
				face = pfaces[i]
				face2 = faces[j]
				face_batch = np.concatenate((face,face2),axis=-1)
				face_batch = face_batch[np.newaxis,...]
				score = predictFS(face_batch,context3,bindings3,d_input3,d_output3,stream3,output3)
				mat_score[i,j] = score 
		# print(mat_score)
		pfaces = faces
		pid = list(range(len(faces)))

		pose_id, temp_ids = get_id_to_assign(pose_id,Q,temp_Ji,mat_score,frame_id)
		# print(temp_ids)

		Q_to_add = []
		for j in range(len(temp_Ji)):
			temp_bbox = temp_bboxes[j]
			temp_j = temp_Ji[j] #local
			temp_v = temp_Vi[j]
			temp_gj = get_global_joints(temp_bbox,temp_j)
			f = faces[j]
			id_to_assign = temp_ids[j]
			temp_pose = new_Pose(frame_id,id_to_assign,temp_j,temp_gj,temp_bbox,temp_v)
			Q_to_add.append(temp_pose)
		frame_id = frame_id + 1
		Q.append(Q_to_add)
	end = timer()

	img_out = np.copy(img_ori)

	if is_draw_skeleton:
		temp_FPS = 1/(end-start)
		total_FPS = total_FPS + end - start
		to_add_str = 'FPS: {:.2f}'.format(temp_FPS)
		pos_FPS = (i_w - 200, i_h - 100)
		c_code = (255,0,0)
		# cv2.putText(img_out,to_add_str,pos_FPS,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
		Qs = Q_to_add
		for qs in Qs:
			tid = qs.id
			tbbox = qs.bbox
			tkps = qs.joints
			tvs = qs.valids

			g_kps = qs.global_joints

			# draw bbox
			c_code = (0,255,0)

			cv2.rectangle(img_out,(int(tbbox[0]),int(tbbox[1])),(int(tbbox[0]+tbbox[2]),int(tbbox[1]+tbbox[3])),c_code,2)

			# put id in image

			pos_txt = (int(tbbox[0]+tbbox[2]-50),int(tbbox[1]+tbbox[3]-10))
			cv2.putText(img_out,str(qs.id),pos_txt,cv2.FONT_HERSHEY_COMPLEX,2,c_code,thickness=2)

			# line
			ci = 0
			for sk in skeleton_list:
				p1,p2 = sk
				x1,y1 = g_kps[p1,:].tolist()
				x2,y2 = g_kps[p2,:].tolist()
				tc = color_list[ci%6]
				ci = ci + 1
				if tvs[p1] == 1 and tvs[p2] == 1:
					cv2.line(img_out,(x1,y1),(x2,y2),tc,2)

			# dot
			for i in range(15):
				x,y = g_kps[i,:].tolist()
				if tvs[i] == 1:
					cv2.circle(img_out,(x,y),2,(0,0,255),3)
	if is_show:
		# print(img_out)
		cv2.imshow('frame', img_out)
	if is_save:
		num_name = '{:07d}'.format(frame_id-1)
		cv2.imwrite('./live_imgs/'+num_name+'.jpg',img_out)
	# if frame_id == 150:
	# 	print(total_FPS)
	# 	print('AVG FPS for 150 frames: {}'.format(150/total_FPS))
	# 	break  
	if not is_show:
		print(frame_id)
		if frame_id == 150:
			print('AVG FPS for 150 frames: {}'.format(150/total_FPS))
			break
	# Waits for a user input to quit the application    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
cap.release()
cv2.destroyAllWindows()