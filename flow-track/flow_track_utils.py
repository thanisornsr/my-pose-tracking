
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
# load human detector
# def load_human_detector(detect_threshold):
# 	cfg = get_cfg()
# 	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
# 	# cfg.merge_from_file(model_zoo.get_config_file("Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml"))
# 	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detect_threshold  # set threshold for this model
# 	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
# 	# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml")
# 	temp_detector = DefaultPredictor(cfg)

# 	return temp_detector
# # load pose estimation model
# def load_pose_es(weight_dir):
# 	gpus = tf.config.experimental.list_physical_devices('GPU')
# 	for gpu in gpus:
# 		tf.config.experimental.set_memory_growth(gpu, True)
# 		limit_temp = tf.config.LogicalDeviceConfiguration(memory_limit=400)
# 		tf.config.experimental.set_virtual_device_configuration(gpu,[limit_temp])
		
# 	def build_model():
# 		# base_model = tf.keras.applications.MobileNetV2(include_top=False, weights = 'imagenet', input_shape = (256,192,3))
# 		base_model = tf.keras.applications.ResNet152(include_top=False, weights = None, input_shape = (256,192,3))
# 		xxx = base_model.output
# 		xxx = layers.Conv2DTranspose(filters= 256,kernel_size= [4,4], strides= 2, padding= 'same',activation='relu')(xxx)
# 		xxx = layers.Conv2DTranspose(filters= 256,kernel_size= [4,4], strides= 2, padding= 'same',activation='relu')(xxx)
# 		xxx = layers.Conv2DTranspose(filters= 256,kernel_size= [4,4], strides= 2, padding= 'same',activation='relu')(xxx)
# 		xxx = layers.Conv2D(filters= 17,kernel_size= [1,1], padding= 'same')(xxx)
# 		model_temp = models.Model(inputs=base_model.input, outputs=xxx)
# 		return model_temp
# 	temp_model = build_model()
# 	temp_model.load_weights(weight_dir)
	# return temp_model

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
		i_bbox = bboxes[i]
		bbox_x, bbox_y,bbox_x2,bbox_y2 = i_bbox
		bbox_w = bbox_x2-bbox_x
		bbox_h = bbox_y2-bbox_y
		# print('Before: {}, {}, {}, {}'.format(bbox_x,bbox_y,bbox_w,bbox_h))
		to_check = 0.75*bbox_h
		if to_check >= bbox_w:
			add_x = True
			# print('Add x')
		else:
			add_x = False
			# print('Add y')


		if add_x:
			new_bbox_h = bbox_h
			new_bbox_w = 0.75*bbox_h
			diff = new_bbox_w - bbox_w
			new_bbox_y = bbox_y
			new_bbox_x = bbox_x - 0.5*diff
			#check if in image
			if new_bbox_x < 0:
				new_bbox_x = 0
			if new_bbox_x+new_bbox_w >= img_w:
				new_bbox_x = img_w - new_bbox_w - 1
		else:
			new_bbox_w = bbox_w
			new_bbox_h = 4.0/3.0 * bbox_w
			diff = new_bbox_h - bbox_h
			new_bbox_x = bbox_x
			new_bbox_y = bbox_y - 0.5 * diff

			if new_bbox_y < 0:
				new_bbox_y = 0
			if new_bbox_y+new_bbox_h >= img_h:
				new_bbox_y = img_h - new_bbox_h - 1
		temp_new_bbox = [new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h]
		# print('After: {}, {}, {}, {}'.format(new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h))
		out_bboxes.append(temp_new_bbox)
	return out_bboxes

# Decode heatmaps to joints
def heatmaps_to_joints(i_heatmaps):
	o_J = []
	valid_thres = 15
	o_V = []
	n_h = i_heatmaps.shape[0]
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

# Calculate similarity matric from OKS
def cal_sim_mat(input_ori_Q,input_Ji,current_frame):
	temp_f = current_frame
	# Use whole Q
	# input_Q = input_ori_Q
	# Use only n-1 frame
	input_Q = [input_ori_Q[x] for x in range(len(input_ori_Q)) if input_ori_Q[x].frame in list(range(temp_f-10,temp_f))]
	OKS_k = np.array([0.026,0.035,0.035,0.079,0.079,0.072,0.072,0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089])
	n_Q = len(input_Q)
	n_J = len(input_Ji)
	sim_mat = np.full([n_J, n_Q], None)
	for ij in range(n_J):
		for iq in range(n_Q):
			i_joint = input_Ji[ij] 
			j_P = input_Q[iq]
			tx1,tx2,tw,th = j_P.bbox
			cal_v = j_P.valids
			s2 = tw*th
			j_joint = j_P.joints
			# calculate score
			d2 = (i_joint - j_joint)**2
			d2 = np.sum(d2,axis=0)
			term =  np.divide(d2,OKS_k**2)
			term = term/(2*s2)
			term_exp = np.exp(-1*term)
			term_exp = np.multiply(term_exp,cal_v)
			temp_OKS = np.sum(term_exp)/np.sum(cal_v)
			if math.isnan(temp_OKS):
				temp_OKS = 0.0
			# assign score to sim_mat
			sim_mat[ij,iq] = temp_OKS

	return sim_mat
# get id list to assign
def get_id_to_assign(inputo_pose_id,input2_ori_Q,input2_Ji,input_sim_mat,current_frame):
	get_id_sim_mat = input_sim_mat
	input_pose_id = inputo_pose_id
	input_nh = len(input2_Ji)
	temp_f = current_frame
	# Use whole Q
	# input2_Q = input2_ori_Q
	# can_old_id = list(range(input_pose_id))

	# Use only n-1 frame
	input2_Q = [input2_ori_Q[x] for x in range(len(input2_ori_Q)) if input2_ori_Q[x].frame in list(range(temp_f-10,temp_f))]
	can_old_id_0 = [input2_Q[x].id for x in range(len(input2_Q))]
	can_old_id = []
	[can_old_id.append(x) for x in can_old_id_0 if x not in can_old_id]

	# print(can_old_id)
	can_new_id = list(range(input_nh))
	id_to_assign = np.full([1,input_nh],None)

	max_score_th = 0.6

	while len(can_old_id)>0 and len(can_new_id)>0 :
		
		# print(id_to_assign)
		get_id_sim_mat = np.nan_to_num(get_id_sim_mat)
		# print(type(get_id_sim_mat))
		max_score = np.amax(get_id_sim_mat)
		# print('Max score: {}'.format(max_score))
		if max_score < max_score_th:
			# print('BREAK!!')
			break
		else:
			res = np.where(get_id_sim_mat == max_score)
			# print('res:{}'.format(res))
			max_pos = [res[0][0],res[1][0]]
			# print(max_pos)

			mi,mj = max_pos
			m_P = input2_Q[mj]
			m_id = m_P.id
			id_to_assign[0,mi] = m_id
			# print(m_id)
			can_old_id.remove(m_id)
			can_new_id.remove(mi)
			column_to_rm = [x for x in range(len(input2_Q)) if input2_Q[x].id == m_id]
			row_to_rm = mi
			get_id_sim_mat[:,column_to_rm] = 0
			get_id_sim_mat[row_to_rm,:] = 0

	n_new = np.sum((id_to_assign == None).astype('int'))
	if n_new > 0:
		for l in range(input_nh):
			id_to_check = id_to_assign[0,l]
			if id_to_check == None:
				id_to_assign[0,l] = input_pose_id
				input_pose_id = input_pose_id + 1
	id_to_assign = id_to_assign.tolist()[0]
	return input_pose_id, id_to_assign
class Pose:
	def __init__(self):
		self.frame = None
		self.id = None
		self.joints = None
		self.bbox = None
		self.valids = None
		self.global_joints = None

def get_global_kps(input_pose,output_shape):
	temp_p = input_pose
	tkps = temp_p.joints
	tbbox = temp_p.bbox

	o_kps = np.transpose(np.copy(tkps))

	if tbbox[2] == 0:
		tbbox[2] = 1
	if tbbox[3] == 0:
		tbbox[3] = 1

	scale_x = tbbox[2]/output_shape[1]
	scale_y = tbbox[3]/output_shape[0]

	temp_scale = np.array([scale_x,scale_y])

	for ikps in range(len(o_kps)):
		temp_val = o_kps[ikps,:]

		o_kps[ikps,:] = np.multiply(temp_val,temp_scale)

	g_kps = o_kps
	o_x = tbbox[0]
	o_y = tbbox[1]
	o_origin = np.array([tbbox[0],tbbox[1]])

	for gkps in range(len(g_kps)):
		if g_kps[gkps,0] != 0:
			g_kps[gkps,:] = g_kps[gkps,:] + o_origin

	temp_p.global_joints = g_kps

	return temp_p


def predict(batch,context2,bindings2,d_input2,d_output2,stream2,output2):
    cuda.memcpy_htod_async(d_input2,batch,stream2)

    context2.execute_async_v2(bindings2,stream2.handle,None)

    cuda.memcpy_dtoh_async(output2, d_output2, stream2)

    stream2.synchronize()

    return output2





def flow_track_from_dir(images_dir,hdetector,context2,bindings2,d_input2,d_output2,stream2,output2,input_shape):
	Q = []
	processing_times = []
	frame_id = 0
	pose_id = 0
	for filename in sorted(glob.glob(images_dir+'/*.jpg')):
		# print('Processing frame {} ...'.format(frame_id))
		start = timer()
		img = cv2.imread(filename)
		
		temp_bboxes = img_to_bboxes(img,hdetector,input_shape)
		img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img_RGB = img_RGB / 255.0
		batch_imgs = []
		# print(temp_bboxes)
		for i_bbox in temp_bboxes:
			# print(i_bbox)
			o_crop = img_RGB[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
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
		# print(p_heatmaps.shape)

		# p_heatmaps = predict(batch_imgs) # TO FIX can predict one by one

		temp_Ji,temp_Vi = heatmaps_to_joints(p_heatmaps)
		# print(temp_Ji)
		# return None,None 

		if frame_id == 0:
			for j in range(nh):
				temp_bbox = temp_bboxes[j]
				temp_j = temp_Ji[j]
				temp_v = temp_Vi[j]
				temp_pose = new_Pose(frame_id,pose_id,temp_j,temp_bbox,temp_v)
				temp_pose = get_global_kps(temp_pose,(64, 48))
				Q.append(temp_pose)
				pose_id = pose_id + 1

			frame_id = frame_id + 1
		else:
			# similarity matrix from Joints
			# print(frame_id)
			temp_sim_mat = cal_sim_mat(Q,temp_Ji,frame_id)
			# print(temp_sim_mat)
			# Assign id and create Pose 
			pose_id, temp_ids = get_id_to_assign(pose_id,Q,temp_Ji,temp_sim_mat,frame_id)
			# print(temp_ids)
			# print('-----------------------------------')
			for j in range(nh):
				temp_bbox = temp_bboxes[j]
				# print(temp_bbox)
				temp_j = temp_Ji[j]
				temp_v = temp_Vi[j]
				t_pose_id = temp_ids[j]
				temp_pose = new_Pose(frame_id,t_pose_id,temp_j,temp_bbox,temp_v)
				temp_pose = get_global_kps(temp_pose,(64, 48))
				# print(temp_pose.bbox)
				Q.append(temp_pose)

			frame_id = frame_id + 1
		end = timer()
		processing_times.append(end-start)

	return Q,processing_times


def clear_output_folder(output_dir):
	for filename in sorted(glob.glob(output_dir+'/*.mp4')):
		os.remove(filename)
	print('Output folder cleared')

def make_video_flow_track(input_images_dir,input_processing_times,input_Q):
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
		temp_FPS = 1/input_processing_times[frame_count]
		to_add_str = 'FPS: {:.2f}'.format(temp_FPS)
		pos_FPS = (i_w - 200, i_h - 100)
		pos_AVG = (i_w - 280, i_h - 150)
		c_code = (255,0,0)
		cv2.putText(img,to_add_str,pos_FPS,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
		cv2.putText(img,to_add_avg,pos_AVG,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
		check_list = [x for x in range(len(input_Q)) if input_Q[x].frame == frame_count]
		Qs = []
		for cl in check_list:
			Qs.append(input_Q[cl])

		for qs in Qs:
			tid = qs.id
			# print(tid)
			tbbox = qs.bbox
			tkps = qs.joints
			tvs = qs.valids

			g_kps = qs.global_joints

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

	#print('Done creating output video')

def write_processing_times_JSON(processing_times,filename):
	pt = {}
	for i in range(len(processing_times)):
		pt[i] = processing_times[i]
	with open(filename,'w') as outfile:
		json.dump(pt,outfile)

def write_Q_JSON(Q,filename):
	data = []
	for q in Q:
		temp_kp = q.global_joints
		temp_v = q.valids
		temp_id = q.id
		temp_bbox = q.bbox
		temp_frame_id = q.frame

		temp_kps = []
		for i in range(len(temp_v)):
			if int(temp_v[i]) == 0:
				temp_kps.append(0)
				temp_kps.append(0)
			else:
				temp_kps.append(int(temp_kp[i,0]))
				temp_kps.append(int(temp_kp[i,1]))
			temp_kps.append(int(temp_v[i]))
		# print(temp_kps)
		temp = {}
		temp['frame_id'] = temp_frame_id
		temp['track_id'] = temp_id
		temp['bbox'] = temp_bbox
		temp['keypoints'] = temp_kps

		data.append(temp)
	with open(filename,'w') as outfile:
		json.dump(data,outfile)






