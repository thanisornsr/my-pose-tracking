
import glob


import numpy as np
import os, json, cv2, random, math

from skimage.transform import resize
from timeit import default_timer as timer

import tensorrt as trt 
import pycuda.driver as cuda

from yolo_classes import get_cls_dict
from yolo_with_plugins import get_input_shape, TrtYOLO

########## functions

def scale_to_43_xywh(input_bbox,input_w,input_h):
	bbox_x, bbox_y,bbox_w,bbox_h = input_bbox
	to_check = 0.75*bbox_h
	if to_check >= bbox_w:
		add_x = True
	else:
		add_x = False

	if add_x:
		new_bbox_h = bbox_h
		new_bbox_w = 0.75*bbox_h
		diff = new_bbox_w - bbox_w
		new_bbox_y = bbox_y
		new_bbox_x = bbox_x - 0.5*diff
		#check if in image
		if new_bbox_x < 0:
			new_bbox_x = 0
		if new_bbox_x+new_bbox_w >= input_w:
			new_bbox_x = input_w - new_bbox_w - 1
	else:
		new_bbox_w = bbox_w
		new_bbox_h = 4.0/3.0 * bbox_w
		diff = new_bbox_h - bbox_h
		new_bbox_x = bbox_x
		new_bbox_y = bbox_y - 0.5 * diff
		if new_bbox_y < 0:
			new_bbox_y = 0
		if new_bbox_y+new_bbox_h >= input_h:
			new_bbox_y = input_h - new_bbox_h - 1

	temp_output_bbox = [new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h]
	return temp_output_bbox

def scale_to_43(input_bbox,input_w,input_h):
	bbox_x, bbox_y,bbox_x2,bbox_y2 = input_bbox
	bbox_w = bbox_x2-bbox_x
	bbox_h = bbox_y2-bbox_y
	to_check = 0.75*bbox_h
	if to_check >= bbox_w:
		add_x = True
	else:
		add_x = False

	if add_x:
		new_bbox_h = bbox_h
		new_bbox_w = 0.75*bbox_h
		diff = new_bbox_w - bbox_w
		new_bbox_y = bbox_y
		new_bbox_x = bbox_x - 0.5*diff
		#check if in image
		if new_bbox_x < 0:
			new_bbox_x = 0
		if new_bbox_x+new_bbox_w >= input_w:
			new_bbox_x = input_w - new_bbox_w - 1
	else:
		new_bbox_w = bbox_w
		new_bbox_h = 4.0/3.0 * bbox_w
		diff = new_bbox_h - bbox_h
		new_bbox_x = bbox_x
		new_bbox_y = bbox_y - 0.5 * diff
		if new_bbox_y < 0:
			new_bbox_y = 0
		if new_bbox_y+new_bbox_h >= input_h:
			new_bbox_y = input_h - new_bbox_h - 1

	temp_output_bbox = [new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h]
	return temp_output_bbox

# get boundary boxes from images:
def img_to_bboxes(img_o,predictor,input_shape):
	img_h,img_w,_ = img_o.shape

	boxes, confs, clss = predictor.detect(img_o)
	boxes = boxes[clss==0]

	boxes = boxes.tolist()
	n_ints = len(boxes)

	out_bboxes = []
	bboxes = boxes
	for i in range(len(bboxes)):
		temp_new_bbox = scale_to_43(bboxes[i],img_w,img_h)
		out_bboxes.append(temp_new_bbox)

	return out_bboxes

# Human detection module: use to detect human in keyframe
def human_detect_module(i_detector,i_img,i_img_RGB,input_shape):
	temp_bboxes = img_to_bboxes(i_img,i_detector,input_shape)
	batch_imgs = []
	for i_bbox in temp_bboxes:
		o_crop = i_img_RGB[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]

		if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
			print('detect empty image')
			continue

		o_crop = resize(o_crop,input_shape)
		o_crop = o_crop.astype('float32')
		batch_imgs.append(o_crop)
	nh = len(batch_imgs)
	# batch_imgs = np.array(batch_imgs)
	# batch_imgs = tf.convert_to_tensor(batch_imgs,dtype=tf.float32)

	return batch_imgs,temp_bboxes,nh

def heatmaps_to_joints(i_heatmaps):
	o_J = [] #joints position
	valid_thres = 15
	o_V = [] #valid
	n_h = p_heatmaps.shape[0]
	for k in range(n_h):
		temp_heatmap = i_heatmaps[k,:,:,:]
		tkp_x = []
		tkp_y = []
		tvs = []
		for j in range(15):
			if np.max(temp_heatmap[:,:,j]) > valid_thres:
				temp_max = np.where(temp_heatmap[:,:,j] == np.amax(temp_heatmap[:,:,j]))
				ty = temp_max[0][0]
				tx = temp_max[1][0]
				tv = 1
			else:
				ty = 0
				tx = 0
				tv = 0

			tkp_x.append(tx)
			tkp_y.append(ty)
			tvs.append(tv)

		tkp = [tkp_x,tkp_y]
		tkp = np.asarray(tkp)
		tvs = np.asarray(tvs)
		o_J.append(tkp)
		o_V.append(tvs)
	return o_J,o_V

# Create new pose
def new_Pose(nf=None,iid=None,jts=None,ibbox=None,iv=None):
	t_new_pose = Pose()
	if nf is not None:
		t_new_pose.frame = nf
	if iid is not None:
		t_new_pose.id = iid
	if jts is not None:
		t_new_pose.joints = jts
	if ibbox is not None:
		t_new_pose.bbox = ibbox
	if iv is not None:
		t_new_pose.valids = iv

	return t_new_pose

class Pose:
	def __init__(self):
		self.frame = None
		self.id = None
		self.joints = None
		self.bbox = None
		self.valids = None

def expand_bboxes(input_bboxes,input_img_h,input_img_w,percent_expand=20):
	# ori_bboxes = input_bboxes
	in_bbox = input_bboxes
	# out_bboxes = []
	ratio_expand = percent_expand / 200
	# thres = 100

	# for in_bbox in ori_bboxes:
	# print(in_bbox)
	ori_x1 = in_bbox[0]
	ori_x2 = in_bbox[0] + in_bbox[2]
	ori_y1 = in_bbox[1]
	ori_y2 = in_bbox[1] + in_bbox[3]

	new_x1 = max((-1*ratio_expand*in_bbox[2])+ori_x1,0)
	new_x2 = min((ratio_expand*in_bbox[2])+ori_x2,input_img_w-1)
	new_y1 = max((-1*ratio_expand*in_bbox[3])+ori_y1,0)
	new_y2 = min((ratio_expand*in_bbox[3])+ori_y2,input_img_h-1)
	new_dx = new_x2-new_x1
	new_dy = new_y2-new_y1
	out_bboxes = [new_x1,new_y1,new_dx,new_dy]
	# out_bboxes.append(new_bbox)
	return out_bboxes

def crop_img_from_bboxes(input_img,input_bboxes):
	ori_img = input_img
	temp_bboxes = input_bboxes
	output_imgs = []
	for i_bbox in temp_bboxes:
		o_crop = ori_img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
		if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
			# print('detect empty image')
			continue

		o_crop = resize(o_crop,input_shape)
		o_crop = o_crop.astype('float32')
		output_imgs.append(o_crop)
	n_out = len(batch_imgs)
	# output_imgs = np.array(output_imgs)
	# output_imgs = tf.convert_to_tensor(output_imgs,dtype=tf.float32)
	return output_imgs,n_out


def get_bbox_from_joints(old_bboxes,predicted_J,predicted_V,output_shape,input_w,input_h,percent_expand):
	output_bboxes = []
	no_joint_idx = []
	output_J = predicted_J
	assert len(old_bboxes) == len(predicted_J)
	for k in range(len(predicted_J)):
		temp_o_j = output_J[k]
		temp_p_v = predicted_V[k]
		temp_old_bbox = old_bboxes[k]
		# print(temp_old_bbox)
		ori_x = temp_old_bbox[0]
		ori_y = temp_old_bbox[1]
		ori_w = temp_old_bbox[2]
		ori_h = temp_old_bbox[3]
		if ori_w == 0:
			ori_w = 1
		if ori_h == 0:
			ori_h = 1

		scale_x = ori_w/output_shape[1]
		scale_y = ori_h/output_shape[0]
		# convert kps to global
		temp_scale = np.array([scale_x,scale_y])
		# print(temp_scale.shape)
		temp_offset = np.array([ori_x,ori_y])
		for ikps in range(temp_o_j.shape[0]):
			temp_val = temp_o_j[ikps,:]
			temp_o_j[ikps,:] = np.multiply(temp_val,temp_scale[ikps]) + temp_offset[ikps]
		# print(temp_o_j)

		# get new bbox
		xs = [temp_o_j[0,x] for x in range(temp_o_j.shape[1]) if temp_p_v[x] != 0]
		ys = [temp_o_j[1,y] for y in range(temp_o_j.shape[1]) if temp_p_v[y] != 0]
		if len(xs) <= 0 and len(ys) <= 0:
			# print('no joint found')
			no_joint_idx.append(k)
			continue
		min_x = min(xs)
		max_x = max(xs)
		min_y = min(ys)
		max_y = max(ys)
		temp_w = max_x - min_x
		temp_h = max_y - min_y
		temp_new_bbox = [min_x,min_y,temp_w,temp_h]
		# print(temp_new_bbox)

		# expand bbox
		temp_new_bbox = expand_bboxes(temp_new_bbox,input_h,input_w,percent_expand=percent_expand)
		# scale bbox to 4:3
		temp_new_bbox = scale_to_43_xywh(temp_new_bbox,input_w,input_h)
		# print(temp_new_bbox)
		output_bboxes.append(temp_new_bbox)
		# convert global kp to local kp
		output_J[k] = temp_o_j
	return output_bboxes,no_joint_idx,output_J

def cal_scoring_matrix(input_Q,input_bboxes):
	lf_Q = input_Q[-1]
	n_Q = len(lf_Q)
	n_J = len(input_bboxes)
	score_mat = np.full([n_J,n_Q],None)
	for ij in range(n_J):
		for iq in range(n_Q):
			j_P = lf_Q[iq]
			c_bbox = input_bboxes[ij]
			old_bbox = j_P.bbox
			bbox_a = [c_bbox[0],c_bbox[1],c_bbox[0]+c_bbox[2],c_bbox[1]+c_bbox[3]] 
			bbox_b = [old_bbox[0],old_bbox[1],old_bbox[0]+old_bbox[2],old_bbox[1]+old_bbox[3]]
			# calculate score
			# get_iou(bbox_a,bbox_b)
			# assign score to sim_mat
			score_mat[ij,iq] = get_iou(bbox_a,bbox_b)
	return score_mat

 def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def get_id_to_assign(inputo_pose_id,input2_ori_Q,input2_Ji,input_sim_mat,current_frame):
	get_id_s_mat = input_sim_mat
	input_pose_id = inputo_pose_id
	input_nh = len(input2_Ji)
	temp_f = current_frame

	input2_Q = input2_ori_Q[-1]
	can_old_id_0 = [input2_Q[x].id for x in range(len(input2_Q))]
	can_old_id = []
	[can_old_id.append(x) for x in can_old_id_0 if x not in can_old_id]

	can_new_id = list(range(input_nh))
	id_to_assign = np.full([1,input_nh],None)

	iou_th = 0.1

	while len(can_old_id)>0 and len(can_new_id)>0:
		max_score = np.amax(get_id_s_mat)
		if max_score < iou_th:
			break
		else:
			res = np.where(get_id_s_mat == max_score)
			max_pos = list(zip(res[0], res[1]))[0]
			mi,mj = max_pos
			m_P = input2_Q[mj]
			m_id = m_P.id
			id_to_assign[0,mi] = m_id
			can_old_id.remove(m_id)
			can_new_id.remove(mi)
			column_to_rm = [x for x in range(len(input2_Q)) if input2_Q[x].id == m_id]
			row_to_rm = mi
			get_id_s_mat[:,column_to_rm] = 0
			get_id_s_mat[row_to_rm,:] = 0

	n_new = np.sum((id_to_assign == None).astype('int'))
	if n_new > 0:
		for l in range(input_nh):
			id_to_check = id_to_assign[0,l]
			if id_to_check == None:
				id_to_assign[0,l] = input_pose_id
				input_pose_id = input_pose_id + 1
	id_to_assign = id_to_assign.tolist()[0]

	return input_pose_id, id_to_assign

def predict(batch,context2,bindings2,d_input2,d_output2,stream2,output2):
    cuda.memcpy_htod_async(d_input2,batch,stream2)

    context2.execute_async_v2(bindings2,stream2.handle,None)

    cuda.memcpy_dtoh_async(output2, d_output2, stream2)

    stream2.synchronize()

    return output2

def light_track_from_dir(images_dir,hdetector,context2,bindings2,d_input2,d_output2,stream2,output2,input_shape,input_key_frame):
# to fix
	Q = []
	processing_times = []

	# tracked_ID = []
	# tracked_status = []

	frame_id = 0
	pose_id = 0
	key_frame = input_key_frame
	for filename in sorted(glob.glob(images_dir+'/*.jpg')):
		start = timer()
		img = cv2.imread(filename)
		img_h,img_w,_ = img.shape

		img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img_RGB = img_RGB / 255.0

		if (frame_id % key_frame) == 0:
			batch_imgs,batch_bboxes,n_human = human_detect_module(hdetector,img,img_RGB,input_shape)
		else:
			last_Q = Q[-1]
			# still_tracking_ID = [tracked_ID[x] for x in range(len(tracked_ID)) if tracked_status[x] == 1]
			still_tracking_ID = [tQ.id for tQ in last_Q]
			bboxes_from_last_frames = [tQ.bbox for tq in last_Q]
			# batch_bboxes = expand_bboxes(bboxes_from_last_frames,img_h,img_w,20)
			batch_imgs,n_human = crop_img_from_bboxes(img_RGB,bboxes_from_last_frames) 

		
		for bimg in batch_imgs:
			img_to_predict = np.reshape(bimg,(1,*input_shape,3))
			tp_heatmaps = predict(img_to_predict,context2,bindings2,d_input2,d_output2,stream2,output2)
			p_heatmaps.append(np.copy(tp_heatmaps[0,:,:,:]))

		p_heatmaps = np.array(p_heatmaps)

		temp_Ji, temp_Vi = heatmaps_to_joints(p_heatmaps)
		# ------------------------------------------------------------------------
		# Done till here
		new_bboxes,no_joint,temp_Ji = get_bbox_from_joints(batch_bboxes,temp_Ji,temp_Vi,output_shape,img_w,img_h,20)
		for idx in no_joint:
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
			Q.append(Q_to_add)
		else:
			t_score_matrix = cal_scoring_matrix(Q,new_bboxes) 
			pose_id, temp_ids = get_id_to_assign(pose_id,Q,temp_Ji,t_score_matrix,frame_id)
			Q_to_add = []
			for j in range(len(temp_ids)):
				temp_pose = new_Pose(frame_id,temp_ids[j],temp_Ji[j],new_bboxes[j],temp_Vi[j])
				Q_to_add.append(temp_pose)
			frame_id = frame_id + 1
			Q.append(Q_to_add)
		end = timer()
		processing_times.append(end-start)

	return Q,processing_times

def clear_output_folder(output_dir):
	for filename in sorted(glob.glob(output_dir+'/*.mp4')):
		os.remove(filename)
	print('Output folder cleared')

def make_video_light_track(input_images_dir,input_processing_times,input_Q):
	img_array = []
	ori_array = []
	avg_time = np.mean(input_processing_times)
	temp_AVG = 1/avg_time
	to_add_avg = 'AVG_FPS: {:.2f}'.format(temp_AVG)
	frame_count = 0

	for filename in sorted(glob.glob(input_images_dir+'/*.jpg')):
		img = cv2.imread(filename)
		i_h, i_w,_ = img.shape
		img_size = (i_w,i_h)
		ori_array.append(img)

		Qs = Q[frame_count]

		temp_FPS = 1/input_processing_times[frame_count]
		to_add_str = 'FPS: {:.2f}'.format(temp_FPS)
		pos_FPS = (i_w - 200, i_h - 100)
		pos_AVG = (i_w - 280, i_h - 150)
		c_code = (255,0,0)
		cv2.putText(img,to_add_str,pos_FPS,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
		cv2.putText(img,to_add_avg,pos_AVG,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)

		for qs in Qs:
			tid = qs.id
			# print(tid)
			tbbox = qs.bbox
			tkps = qs.joints # already global
			tvs = qs.valids

			g_kps = np.transpose(tkps)

			
			# draw bbox
			c_code = (0,255,0)

			cv2.rectangle(img,(int(tbbox[0]),int(tbbox[1])),(int(tbbox[0]+tbbox[2]),int(tbbox[1]+tbbox[3])),c_code,3)

			# put id in image

			pos_txt = (int(tbbox[0]+tbbox[2]-50),int(tbbox[1]+tbbox[3]-10))
			cv2.putText(img,str(qs.id),pos_txt,cv2.FONT_HERSHEY_COMPLEX,2,c_code,thickness=3)

			# draw skeleton
			skeleton_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
			color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

			# line
			ci = 0
			for sk in skeleton_list:
				p1,p2 = sk
				x1,y1 = g_kps[p1,:].tolist()
				x2,y2 = g_kps[p2,:].tolist()
				tc = color_list[ci%6]
				ci = ci + 1
				if tvs[p1] == 1 and tvs[p2] == 1:
					cv2.line(img,(x1,y1),(x2,y2),tc,3)

			# dot
			for i in range(15):
				x,y = g_kps[i,:].tolist()
				if tvs[i] == 1:
					cv2.circle(img,(x,y),2,(0,0,255),4)

		img_array.append(img)
		frame_count = frame_count + 1

	# out = cv2.VideoWriter('./output/original.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, img_size)
	# for i in range(len(ori_array)):
	# 	out.write(ori_array[i])
	# out.release()
	four_cc = cv2.VideoWriter_fourcc(*'mp4v')

	out = cv2.VideoWriter('output.mp4',four_cc, 15, img_size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()