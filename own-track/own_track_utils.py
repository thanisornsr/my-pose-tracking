import glob
import numpy as np 
import os, json, cv2, random, math

from skimage.transform import resize
from skimage.color import rgb2gray
from timeit import default_timer as timer 

import tensorrt as trt 
import pycuda.driver as cuda

from yolo_classes import get_cls_dict
from yolo_with_plugins import get_input_shape, TrtYOLO



class Pose:
	def __init__(self):
		self.frame = None
		self.id = None
		self.joints = None
		self.global_joints = None
		self.bbox = None
		self.face = None
		self.valids = None

# Create new pose
def new_Pose(nf=None,iid=None,jts=None,gjts=None,ibbox=None,iv=None,iface=None):
	t_new_pose = Pose()
	if nf is not None:
		t_new_pose.frame = nf
	if iid is not None:
		t_new_pose.id = iid
	if jts is not None:
		t_new_pose.joints = jts
	if gjts is not None:
		t_new_pose.global_joints = gjts
	if ibbox is not None:
		t_new_pose.bbox = ibbox
	if iv is not None:
		t_new_pose.valids = iv
	if iface is not None:
		t_new_pose.face = iface
	return t_new_pose

# get boundary boxes from images:
def img_to_bboxes(img_o,predictor,img_h,img_w):
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
		# print(temp_new_bbox)
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
		# print(tkp_x)
		# print(tkp_y)
		# print(tvs)

		tkp = [tkp_x,tkp_y]
		tkp = np.asarray(tkp)
		tvs = np.asarray(tvs)
		o_J.append(tkp)
		o_V.append(tvs)
		
	return o_J,o_V

def get_global_joints(input_bbox,input_joints,output_shape = (64,48)):
	scale_x = input_bbox[2]/output_shape[1]
	scale_y = input_bbox[3]/output_shape[0]
	o_scale = np.array([scale_x,scale_y])

	g_kps = np.transpose(input_joints.copy())
	o_x = input_bbox[0]
	o_y = input_bbox[1]
	o_origin = np.array([o_x,o_y])

	for ikps in range(len(g_kps)):
		g_kps[ikps,:] = np.multiply(g_kps[ikps,:],o_scale) + o_origin

	return g_kps

def get_face_from_joints(gray_img,input_joint,head_to_width_ratio = 0.5):
	no_head = False
	if input_joint[0,0] == 0 and input_joint[0,1] == 0:
		# print(input_joint)
		# print('No head detected')
		if input_joint[1,0] != 0 and input_joint[1,1] != 0 and input_joint[2,0] != 0 and input_joint[2,1] != 0:
			# found both ear
			c_x = (input_joint[1,0] + input_joint[2,0]) / 2
			c_y = (input_joint[1,1] + input_joint[2,1]) / 2
		else:
			if input_joint[1,0] != 0 and input_joint[1,1] != 0:
				#found l ear
				c_x = input_joint[1,0]
				c_y = input_joint[1,1]

			else:
				if input_joint[2,0] != 0 and input_joint[2,1] != 0:
					#found r ear
					c_x = input_joint[2,0]
					c_y = input_joint[2,1]
				else:
					# no nose and no ears found
					# print(input_joint)
					# print('No head detected')
					no_head = True

	else:
		# nose found
		# c: center of head
		c_x = input_joint[0,0]
		c_y = input_joint[0,1]

	if no_head:
		cropped = np.ones((40,40,1)).astype(np.float32)
		# print(cropped.dtype)

	else:
		ih,iw,_ = gray_img.shape
		head_size = head_to_width_ratio * iw
		# print(head_size)
		if head_size <= 0:
			head_size = 60

		x1 = c_x - (head_size / 2)
		x2 = c_x + (head_size / 2)
		y1 = c_y - (head_size / 2)
		y2 = c_y + (head_size / 2)

		if x1 < 0:
			x1 = 0
		if y1 < 0:
			y1 = 0
		if x2 >= iw:
			x2 = iw - 1
		if y2 >= ih:
			y2 = ih - 1
		if (int(x2) - int(x1)) < 1:
			x1 = int(x2) - 1
		if (int(y2) - int(y1)) < 1:
			y1 = int(y2) - 1

		cropped = gray_img[int(y1):int(y2),int(x1):int(x2),:]
		# print(cropped.dtype)

	return cropped

def get_id_to_assign(input_pose_id,input2_ori_Q,input2_Ji,get_id_sim_mat,current_frame):
	input_nh = len(input2_Ji)

	temp_f = current_frame

	input2_Q = input2_ori_Q[-1]
	# print(input2_Q)
	can_old_id_0 = [input2_Q[x].id for x in range(len(input2_Q))]
	can_old_id = list(set(can_old_id_0))

	can_new_id = list(range(input_nh))
	id_to_assign = np.full([1,input_nh],None)
	# print(get_id_sim_mat)

	max_score_th = 0.01
	max_score_th = 0.1
	while len(can_old_id)>0 and len(can_new_id)>0:
		max_score = np.amax(get_id_sim_mat)
		if max_score < max_score_th:
			break
		else:
			res = np.where(get_id_sim_mat == np.amax(get_id_sim_mat))
			max_pos = list(zip(res[0], res[1]))[0]
			mi,mj = max_pos
			m_P = input2_Q[mi]
			m_id = m_P.id
			id_to_assign[0,mj] = m_id
			can_old_id.remove(m_id)
			can_new_id.remove(mj)
			# column_to_rm = [x for x in range(len(input2_Q)) if input2_Q[x].id == m_id]
			column_to_rm = mj
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

def predict(batch,context2,bindings2,d_input2,d_output2,stream2,output2):
	cuda.memcpy_htod_async(d_input2,batch,stream2)
	
	context2.execute_async_v2(bindings2,stream2.handle,None)
	
	cuda.memcpy_dtoh_async(output2, d_output2, stream2)
	
	stream2.synchronize()
	
	return output2
	
def predictFS(batch3,context3,bindings3,d_input3,d_output3,stream3,output3):
    cuda.memcpy_htod_async(d_input3,batch3,stream3)

    context3.execute_async_v2(bindings3,stream3.handle,None)

    cuda.memcpy_dtoh_async(output3, d_output3, stream3)

    stream3.synchronize()

    return output3

def own_track_from_dir(images_dir,hdetector,context2,bindings2,d_input2,d_output2,stream2,output2,input_shape,context3,bindings3,d_input3,d_output3,stream3,output3):
	Q = []
	processing_times = []
	frame_id = 0

	pose_id = 0

	for filename in sorted(glob.glob(images_dir+'/*.jpg')):
		print(frame_id)
		start = timer()
		img = cv2.imread(filename)
		img_h,img_w,_ = img.shape

		temp_bboxes = img_to_bboxes(img,hdetector,img_h,img_w)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
		processing_times.append(end-start)

	return Q,processing_times

def clear_output_folder(output_dir):
	for filename in sorted(glob.glob(output_dir+'/*.mp4')):
		os.remove(filename)
	print('Output folder cleared')

def make_video_own_track(input_images_dir,input_processing_times,Q,output_name):
	img_array = []
	# ori_array = []
	avg_time = np.mean(input_processing_times)
	temp_AVG = 1/avg_time
	to_add_avg = 'AVG_FPS: {:.2f}'.format(temp_AVG)
	frame_count = 0

	for filename in sorted(glob.glob(input_images_dir+'/*.jpg')):
		print(filename)
		img = cv2.imread(filename)
		i_h, i_w,_ = img.shape
		img_size = (i_w,i_h)
		# ori_array.append(img)
		temp_FPS = 1/input_processing_times[frame_count]
		to_add_str = 'FPS: {:.2f}'.format(temp_FPS)
		pos_FPS = (i_w - 200, i_h - 100)
		pos_AVG = (i_w - 280, i_h - 150)
		c_code = (255,0,0)
		cv2.putText(img,to_add_str,pos_FPS,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
		cv2.putText(img,to_add_avg,pos_AVG,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
		# check_list = [x for x in range(len(input_Q)) if input_Q[x].frame == frame_count]
		# Qs = []
		# for cl in check_list:
		# 	Qs.append(input_Q[cl])
		Qs = Q[frame_count]

		for qs in Qs:
			tid = qs.id
			# print(tid)
			tbbox = qs.bbox
			tkps = qs.joints
			tvs = qs.valids

			g_kps = qs.global_joints

			# draw bbox
			c_code = (0,255,0)

			cv2.rectangle(img,(int(tbbox[0]),int(tbbox[1])),(int(tbbox[0]+tbbox[2]),int(tbbox[1]+tbbox[3])),c_code,2)

			# put id in image

			pos_txt = (int(tbbox[0]+tbbox[2]-50),int(tbbox[1]+tbbox[3]-10))
			cv2.putText(img,str(qs.id),pos_txt,cv2.FONT_HERSHEY_COMPLEX,2,c_code,thickness=2)

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
					cv2.line(img,(x1,y1),(x2,y2),tc,2)

			# dot
			for i in range(15):
				x,y = g_kps[i,:].tolist()
				if tvs[i] == 1:
					cv2.circle(img,(x,y),2,(0,0,255),3)
		# img_out_name = './temp/{0:05d}.jpg'.format(frame_count)
		# cv2.imwrite(img_out_name,img)
		img_array.append(img)
		frame_count = frame_count + 1

	
	four_cc = cv2.VideoWriter_fourcc(*'mp4v')

	out = cv2.VideoWriter(output_name,four_cc, 8, img_size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

def write_processing_times_JSON_OT(processing_times,filename):
	pt = {}
	for i in range(len(processing_times)):
		pt[i] = processing_times[i]
	with open(filename,'w') as outfile:
		json.dump(pt,outfile)

def write_Q_JSON_OT(Q,filename):
	data = []
	for qf in Q: 
		for q in qf:
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
			temp = {}
			temp['frame_id'] = temp_frame_id
			temp['track_id'] = temp_id
			temp['bbox'] = temp_bbox
			temp['keypoints'] = temp_kps

			data.append(temp)
	with open(filename,'w') as outfile:
		json.dump(data,outfile)