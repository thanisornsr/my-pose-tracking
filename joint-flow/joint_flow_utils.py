import glob

import numpy as np
import os, json, cv2, random, math

import scipy 
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from skimage.transform import resize
from timeit import default_timer as timer

import tensorrt as trt 
import pycuda.driver as cuda

# from yolo_classes import get_cls_dict
# from yolo_with_plugins import get_input_shape, TrtYOLO

########## functions

def get_peak_dict(input_heatmaps):
	co_dict = {}
	neighborhood_size = 5
	threshold = 0.5
	for i in range(input_heatmaps.shape[-1]):
		filtered_co = []
		data = input_heatmaps[:,:,i]
		data_max = filters.maximum_filter(data, neighborhood_size)

		maxima = (data == data_max)
		data_min = filters.minimum_filter(data, neighborhood_size)
		diff = ((data_max - data_min) > threshold)
		maxima[diff == 0] = 0

		labeled, num_objects = ndimage.label(maxima)
		slices = ndimage.find_objects(labeled)
		x, y = [], []
		for dy,dx in slices:
			x_center = (dx.start + dx.stop - 1)/2
			x.append(int(x_center))
			y_center = (dy.start + dy.stop - 1)/2
			y.append(int(y_center))
		filtered_co = [[y[t],x[t]] for t in range(len(x))]
		co_dict[i] = filtered_co
	return co_dict
# ---------------------------------------------------------------------------------------------------------------------------

# def plot_peak(input_img,input_peak_dict):
# 	input_shape = 368
# 	output_shape = 46
# 	plt.imshow(input_img)
# 	for idx,peak in input_peak_dict.items():
# 		temp_p = np.array(peak)*input_shape / output_shape
# 		if temp_p.size != 0:
# 			plt.plot(temp_p[:,1],temp_p[:,0],'r.')
# 	plt.show()

# ---------------------------------------------------------------------------------------------------------------------------
def get_score_mat_dict(input_peak_dict,input_paf):
	limb_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
	limb_score_dict = {}
	for limb_idx in range(len(limb_list)):
		temp_limb = limb_list[limb_idx]
		paf_x = input_paf[:,:,limb_idx]
		paf_y = input_paf[:,:,limb_idx+16]
		# print(temp_limb)
		id_A = temp_limb[0]
		id_B = temp_limb[1]
		candidate_A = input_peak_dict[id_A]
		candidate_B = input_peak_dict[id_B]

		score_mat = np.ones((len(candidate_A),len(candidate_B))) * np.NaN
		for i in range(len(candidate_A)):
			for j in range(len(candidate_B)):
				temp_A = np.array(candidate_A[i])
				temp_B = np.array(candidate_B[j])

				score_mat[i,j] = cal_paf_score(temp_A,temp_B,paf_x,paf_y)

		score_mat = np.nan_to_num(score_mat,nan=0)
		limb_score_dict[limb_idx] = score_mat
	return limb_score_dict

# ---------------------------------------------------------------------------------------------------------------------------

def cal_paf_score(input_p0,input_p1,input_paf_x,input_paf_y):
	norm = np.linalg.norm(input_p1-input_p0)
	if norm == 0:
	# print('limb is too short')
		return 0
	v = (input_p1 - input_p0) / norm
	t_v = np.array([v[1],v[0]])
	v = t_v
	vt = np.transpose(v)
	n_os = 20
	step_size = 1.0/(n_os-1)
	os = np.arange(0.0,1.0,step_size)
	os = np.append(os,1.0)
	os2 = np.array([[os],[os]]).reshape((2,n_os))

	p0_1 = np.dstack([input_p0]*n_os).reshape((2,n_os))
	p1_1 = np.dstack([input_p1]*n_os).reshape((2,n_os))

	ios = np.around(np.multiply((1-os2),p0_1) + np.multiply(os2,p1_1)).astype('int')

	T = np.zeros(ios.shape)
	for num in range(ios.shape[-1]):
		ii = ios[0,num]
		jj = ios[1,num]

		v_x = input_paf_x[ii,jj]
		v_y = input_paf_y[ii,jj]

		temp_v = np.array([v_x,v_y])
		temp_norm = np.linalg.norm(temp_v)

		if temp_norm > 0:
			temp_v = temp_v/temp_norm
		T[:,num] = temp_v
	scores = []
	for ii in range(n_os):
		t1 = T[:,ii]
		t_score = t1 @ vt
		scores.append(t_score)

	f_score = np.array(scores).mean()
	return f_score

# ---------------------------------------------------------------------------------------------------------------------------

def get_limb_dict(input_score_dict,input_peak_dict):
	limb_dict = {}
	min_thres = 0.55
	limb_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]

	for i in range(len(input_score_dict.keys())):
		temp_limb_list = []
		temp_limb = limb_list[i]
		temp_score = np.copy(input_score_dict[i])
		list_idx_A = list(range(temp_score.shape[0]))
		list_idx_B = list(range(temp_score.shape[1]))

		while len(list_idx_A)>0 and len(list_idx_B)>0:
			temp_max = np.amax(temp_score)
			if temp_max < min_thres:
				break
			else:
				res = np.where(temp_score == temp_max)
				max_pos = list(zip(res[0],res[1]))[0]
				mi,mj = max_pos
				temp_peak_A = input_peak_dict[temp_limb[0]][mi]
				temp_peak_B = input_peak_dict[temp_limb[1]][mj]
				temp_limb_list.append([temp_peak_A,temp_peak_B])
				list_idx_A.remove(mi)
				list_idx_B.remove(mj)

				temp_score[:,mj] = 0
				temp_score[mi,:] = 0
		limb_dict[i] = temp_limb_list
	return limb_dict

# ---------------------------------------------------------------------------------------------------------------------------

class Pose:
	def __init__(self):
		self.frame = None
		self.id = None
		self.joints = {}
		self.global_joints = {}
		self.limbs = {}
		for js in range(15):
			self.joints[js] = [-10000,-10000]
		for ls in range(16):
			self.limbs[ls] = None
		self.bbox = None
		self.valids = None

# ---------------------------------------------------------------------------------------------------------------------------
def poses_from_limb_dict(temp_id,input_limb_dict):
	poses = []
	limb_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
	limb_list_order = [8,9,10,11,4,5,6,7,12,13,14,15,2,3,0,1]
	temp_limb_dict = input_limb_dict
	# print(temp_limb_dict)
	for i in range(len(limb_list_order)):
		temp_order = limb_list_order[i]
		limb_idx = limb_list[temp_order]
		joint_A = limb_idx[0]
		joint_B = limb_idx[1]
		# print(limb_idx)
		for j in range(len(temp_limb_dict[temp_order])):
			temp_limb = temp_limb_dict[temp_order][j]
			if temp_id == 0:
			# print('New Pose {}'.format(temp_id))
				temp_Pose = Pose()
				temp_Pose.limbs[temp_order] = temp_limb
				temp_Pose.joints[joint_A] = temp_limb[0]
				temp_Pose.joints[joint_B] = temp_limb[1]
				temp_Pose.id = temp_id
				poses.append(temp_Pose)
				temp_id = temp_id + 1
			else:
				can_joint_A = temp_limb[0]
				can_joint_B = temp_limb[1]
				new_candidate = True
				for P in poses:
					if (P.joints[joint_A] == can_joint_A) or (P.joints[joint_B] == can_joint_B):
						new_candidate = False
						old_id = P.id
						continue

				if new_candidate:
					temp_Pose = Pose()
					temp_Pose.limbs[temp_order] = temp_limb
					temp_Pose.joints[joint_A] = can_joint_A
					temp_Pose.joints[joint_B] = can_joint_B
					temp_Pose.id = temp_id
					poses.append(temp_Pose)
					temp_id = temp_id + 1
				else:
					old_id_idx = [x for x in range(len(poses)) if poses[x].id == old_id]
					old_id_idx = old_id_idx[0]
					poses[old_id_idx].limbs[temp_order] = temp_limb
					poses[old_id_idx].joints[joint_A] = can_joint_A
					poses[old_id_idx].joints[joint_B] = can_joint_B
	return poses,temp_id

# ---------------------------------------------------------------------------------------------------------------------------

def clean_poses(input_poses):
	cleaned_poses = []
	for P in input_poses:
		temp_joints = P.joints.copy()
		len_joints = len(P.joints.keys())
		for key,value in P.joints.items():
			if value == [-10000,-10000]:
				temp_joints.pop(key)
		P.joints = temp_joints
		len_joints = len(temp_joints.keys())
		# print(len_joints)
		if len_joints >= 4:
			cleaned_poses.append(P)
	return cleaned_poses

# ---------------------------------------------------------------------------------------------------------------------------
# def show_skeleton(input_img,input_cleaned,is_show_id = False):
# 	limb_list = [(0,1),(2,0),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
# 	color_list = ['g','b','r','c','m','y']
# 	ci = 0
# 	for P in input_cleaned:
# 		tc = color_list[P.id % 6]
# 		# tc = color_list[ci%6]
# 		ci = ci + 1
# 		for sk in limb_list:
# 			p1,p2 = sk
# 			k_list = P.joints.keys()
# 			if p1 not in k_list or  p2 not in k_list:
# 				continue
# 			x1 = P.joints[p1][1]*8
# 			y1 = P.joints[p1][0]*8
# 			x2 = P.joints[p2][1]*8
# 			y2 = P.joints[p2][0]*8
# 		if (x1 >= 0) and (x2 >= 0) and (y1 >= 0) and (y2 >= 0):
# 			plt.plot([x1,x2],[y1,y2], color = tc, linewidth = 3)
# 	plt.imshow(input_img)
# 	plt.show()
# ---------------------------------------------------------------------------------------------------------------------------

def cal_T_score(input_p0,input_p1,input_TFF_x,input_TFF_y):
	norm = np.linalg.norm(input_p1-input_p0)
	v = (input_p1-input_p0) / norm
	vt = np.transpose(v)

	n_os = 20
	step_size = 1.0/(n_os-1)
	os = np.arange(0.0,1.0,step_size)
	os = np.append(os,1.0)

	os2 = np.array([[os],[os]]).reshape((2,n_os))

	p0_1 = np.dstack([input_p0]*n_os).reshape((2,n_os))
	p1_1 = np.dstack([input_p1]*n_os).reshape((2,n_os))

	ios = np.around(np.multiply((1-os2),p0_1) + np.multiply(os2,p1_1)).astype('int')

	T = np.zeros(ios.shape)

	for num in range(ios.shape[-1]):
		i = ios[1,num]
		j = ios[0,num]
		# print('{},{}'.format(i,j))
		v_x = input_TFF_x[i,j]
		v_y = input_TFF_y[i,j]
		temp_v = np.array([v_x,v_y])
		temp_norm = np.linalg.norm(temp_v)
		if temp_norm > 0:
			temp_v = temp_v/temp_norm
		# print('{},{}'.format(v_x,v_y))
		T[:,num] = temp_v
	scores = []
	for i in range(n_os):
		t1 = T[:,i]
		t_score =  t1 @ vt
		scores.append(t_score)
	f_score = np.array(scores).mean()
	return f_score

# ---------------------------------------------------------------------------------------------------------------------------

def get_TFF_score_mat(input_ps0,input_ps1,input_TFF):
	e_thres = 2
	n_ps0 = len(input_ps0)
	n_ps1 = len(input_ps1)
	score_mat = np.zeros((n_ps0,n_ps1))
	for i_old in range(n_ps0):
		for i_new in range(n_ps1):
			p0 = input_ps0[i_old]
			p1 = input_ps1[i_new]
			p0_j = p0.joints
			p1_j = p1.joints
			mutual_j = [x for x in p0_j.keys() if x in p1_j.keys()]
			kp_scores = []
			for mj in mutual_j:
				kp0 = np.array(p0_j[mj])
				kp1 = np.array(p1_j[mj])
				ed = np.linalg.norm(kp1-kp0)
				if ed <= e_thres:
					# print('not moving')
					kp_score = 1
				else:
					# print('moving')
					kp_score = cal_T_score(kp0,kp1,input_TFF[:,:,mj],input_TFF[:,:,mj+15])
				kp_scores.append(kp_score)
				final_score = np.sum(np.array(kp_scores))
				score_mat[i_old,i_new] = final_score

	return score_mat

# ---------------------------------------------------------------------------------------------------------------------------
def assign_id_to_current_frame(input_p0,input_p1,last_id,input_score_mat):
	can_old_id = [x.id for x in input_p0]
	can_new_id = list(range(len(input_p1)))
	out_p = input_p1

	score_mat = np.copy(input_score_mat)
	score_th = 1
	while len(can_old_id) > 0 and len(can_new_id) > 0:
		max_score = np.amax(score_mat)
		if max_score < score_th:
			break
		else:
			res = np.where(score_mat == max_score)
			max_pos = list(zip(res[0],res[1]))[0]
			mi,mj = max_pos
			m_id = input_p0[mi].id
			out_p[mj].id = m_id

			can_old_id.remove(m_id)
			can_new_id.remove(mj)

			score_mat[:,mj] = 0
			score_mat[mi,:] = 0

	for id in can_new_id:
		out_p[id].id = last_id
		last_id = last_id + 1

	return out_p,last_id

# ---------------------------------------------------------------------------------------------------------------------------

def set_valid_to_poses(input_poses):
	temp_poses = input_poses
	for i in range(len(temp_poses)):
		temp_pose = temp_poses[i]
		joint_idx = list(temp_pose.joints.keys())
		temp_valids = np.zeros(15)
		temp_valids[joint_idx] = 1
		temp_poses[i].valids = temp_valids

	return temp_poses

# ---------------------------------------------------------------------------------------------------------------------------

def convert_kps_to_global(input_poses,img_w,img_h,output_shape = (46,46)):
	temp_poses = input_poses
	for temp_pose in temp_poses:
		keys = list(temp_pose.joints.keys())
		# print('{},{}'.format(img_h,img_w))
		scale_x = img_w/output_shape[0]
		scale_y = img_h/output_shape[0]
		# print('{},{}'.format(scale_x,scale_y))
		for k in keys:
			temp = temp_pose.joints[k]
			tx = int(temp[1] * scale_x)
			ty = int(temp[0] * scale_y)
			temp_pose.global_joints[k] = [ty,tx]
	return temp_poses

# ---------------------------------------------------------------------------------------------------------------------------

def joint_flow_pipeline_first_frame(input_pose_id,input_img,context2,bindings2,d_input2,d_output2,stream2,output2):
	prediction = predict_open(input_img,context2,bindings2,d_input2,d_output2,stream2,output2)

	p_bms = prediction[:,:,:,128:143]
	p_pafs = prediction[:,:,:,143:]

	p_bm = p_bms[0,:,:,:]
	p_paf = p_pafs[0,:,:,:]

	p_peaks = get_peak_dict(p_bm)
	p_score_dict = get_score_mat_dict(p_peaks,p_paf)
	p_limb_dict = get_limb_dict(p_score_dict,p_peaks)
	p_output_poses,output_pose_id = poses_from_limb_dict(input_pose_id,p_limb_dict)
	p_cleaned_output_poses = clean_poses(p_output_poses)

	return p_cleaned_output_poses, output_pose_id
# ---------------------------------------------------------------------------------------------------------------------------
def joint_flow_pipeline(input_pose_id,input_img0,input_img1,input_poses0,context3,bindings3,d_input3,d_output3,stream3,output3):
	inp_batch = np.concatenate((input_img0,input_img1),axis=-1)
	prediction = predict_TFF(inp_batch,context3,bindings3,d_input3,d_output3,stream3,output3)
	p_TFF = prediction[0,:,:,0:30]
	p_bm1 = prediction[0,:,:,45:60]
	p_paf1 = prediction[0,:,:,92:]

	p_peaks1 = get_peak_dict(p_bm1)
	p_score_dict1 = get_score_mat_dict(p_peaks1,p_paf1)
	p_limb_dict1 = get_limb_dict(p_score_dict1,p_peaks1)
	p_output_poses1, _ = poses_from_limb_dict(0,p_limb_dict1)
	p_cleaned_output_poses1 = clean_poses(p_output_poses1)

	p_score_mat = get_TFF_score_mat(input_poses0,p_cleaned_output_poses1,p_TFF)
	output_poses, output_pose_id = assign_id_to_current_frame(input_poses0,p_cleaned_output_poses1,input_pose_id,p_score_mat)
	return output_poses, output_pose_id

# ---------------------------------------------------------------------------------------------------------------------------
def clear_output_folder(output_dir):
	for filename in sorted(glob.glob(output_dir+'/*.mp4')):
		os.remove(filename)
	print('Output folder cleared')
# ---------------------------------------------------------------------------------------------------------------------------
def make_vid_from_dict_joint_flow(Q,input_times,input_img_dir):
	img_array = []
	ori_array = []
	frame_count = 0
	avg_time = np.mean(list(input_times.values()))
	temp_AVG = 1/avg_time
	to_add_avg = 'AVG_FPS: {:.2f}'.format(temp_AVG)

	for filename in sorted(glob.glob(input_img_dir+'/*.jpg')):
		img = cv2.imread(filename)
		if frame_count == 0:
			i_h,i_w,i_l = img.shape
			img_size = (i_w,i_h)
			# print(img_size)
		ori_array.append(img)
		Qs = Q[frame_count]

		temp_FPS = 1/input_times[frame_count]
		to_add_str = 'FPS: {:.2f}'.format(temp_FPS)
		c_code = (255,0,0)
		pos_FPS = (i_w-200,i_h-100)
		pos_AVG = (i_w-280,i_h-150)
		cv2.putText(img,to_add_str,pos_FPS,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)
		cv2.putText(img,to_add_avg,pos_AVG,cv2.FONT_HERSHEY_COMPLEX,1,c_code,thickness=3)

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
			cv2.putText(img,str(qs.id),pos_txt,cv2.FONT_HERSHEY_COMPLEX,2,c_code,thickness=3)

			#draw skeleton
			skeleton_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
			color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
			# line
			ci = 0
			for sk in skeleton_list:
				p1,p2 = sk
				if tvs[p1] == 1 and tvs[p2] == 1:
					y1,x1 = tkps[p1]
					y2,x2 = tkps[p2]
					tc = color_list[ci%6]
					cv2.line(img,(x1,y1),(x2,y2),tc,3)
				ci = ci + 1
			# dot
			for i in range(len(tvs)):
				if tvs[i] == 1:
					y,x = tkps[i]
					cv2.circle(img,(x,y),2,(0,0,255),4)
		img_array.append(img)
		frame_count = frame_count + 1

	# out = cv2.VideoWriter('original.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, img_size)
	# for i in range(len(ori_array)):
	# 	out.write(ori_array[i])
	# out.release()
	four_cc = cv2.VideoWriter_fourcc(*'mp4v')

	out = cv2.VideoWriter('output.mp4',four_cc, 15, img_size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

# ---------------------------------------------------------------------------------------------------------------------------
def predict_open(batch,context2,bindings2,d_input2,d_output2,stream2,output2):
    cuda.memcpy_htod_async(d_input2,batch,stream2)
    context2.execute_async_v2(bindings2,stream2.handle,None)
    cuda.memcpy_dtoh_async(output2fff, d_output2fff, stream2)
    # cuda.memcpy_dtoh_async(output2bm, d_output2bm, stream2)
    # cuda.memcpy_dtoh_async(output2paf, d_output2paf, stream2)

    stream2.synchronize()

    return output2

def predict_TFF(batch0,context3,bindings3,d_input3,d_output3,stream3,output3):
	cuda.memcpy_htod_async(d_input3,batch0,stream3)
	# cuda.memcpy_htod_async(d_input31,batch1,stream3)
	context3.execute_async_v2(bindings3,stream3.handle,None)
	cuda.memcpy_dtoh_async(output3, d_output3, stream3)
	# cuda.memcpy_dtoh_async(output3bm0, d_output3bm0, stream3)
	# cuda.memcpy_dtoh_async(output3bm1, d_output3bm1, stream3)
	# cuda.memcpy_dtoh_async(output3paf0, d_output3paf0, stream3)
	# cuda.memcpy_dtoh_async(output3paf1, d_output3paf1, stream3)
	stream3.synchronize()
	return output3
# ---------------------------------------------------------------------------------------------------------------------------

def joint_flow_from_dir(images_dir,input_shape,output_shape,context2,bindings2,d_input2,d_output2,stream2,output2,context3,bindings3,d_input3,d_output3,stream3,output3):
	Q = {}
	processing_times = {}
	frame_id = 0
	pose_id = 0

	for filename in sorted(glob.glob(images_dir+'/*.jpg')):
		start = timer()
		img = cv2.imread(filename)
		img_h, img_w, _ = img.shape
		# print(img.shape)

		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img = img / 255.0

		img = np.reshape(resize(img,input_shape),(1,*input_shape,-1))
		
		img = img.astype(np.float32)
		# print(img.shape)
		# print(img.dtype)
		if frame_id == 0:
			poses,pose_id = joint_flow_pipeline_first_frame(pose_id,img,context2,bindings2,d_input2,d_output2,stream2,output2)
			poses = set_valid_to_poses(poses)
			poses = convert_kps_to_global(poses,img_w,img_h)

			Q[frame_id] = poses
			poses0 = poses
			img0 = img 
			end = timer()
			processing_times[frame_id] = end - start
			frame_id = frame_id + 1
		else:
			poses,pose_id = joint_flow_pipeline(pose_id,img0,img,poses0,context3,bindings3,d_input3,d_output3,stream3,output3)
			poses = set_valid_to_poses(poses)
			poses = convert_kps_to_global(poses,img_w,img_h)

			Q[frame_id] = poses
			poses0 = poses
			img0 = img 
			end = timer()
			processing_times[frame_id] = end - start
			frame_id = frame_id + 1

	return Q,processing_times

# ---------------------------------------------------------------------------------------------------------------------------
