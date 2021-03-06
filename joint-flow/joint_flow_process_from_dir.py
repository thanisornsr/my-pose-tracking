import argparse
from joint_flow_utils import *
import glob
import numpy as np
# from yolo_classes import get_cls_dict
# from yolo_with_plugins import get_input_shape, TrtYOLO

import cv2
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit

def read_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('input_dir')

	args = parser.parse_args()

	return args 

a = read_arguments()

print('Input folder directory: {}'.format(a.input_dir))

# config 
# myBatchSize = 1
BATCH_SIZE = 1
# myInputShape = (368,368)
input_shape = (368,368)
# myOutputShape = (46,46)
output_shape = (46,46)
n_keypoints = 15
n_limbs = 16
argument_list = a.input_dir.split('/')
USE_FP16 = False # USE FP32

# build openpose
print('Generating Openpose: ...')

dummy_input_batch = np.zeros((BATCH_SIZE,*input_shape,3),dtype = np.float32)

target_dtype = np.float16 if USE_FP16 else np.float32

f2 = open("open_pose_engine.trt","rb")
runtime2 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine2 = runtime2.deserialize_cuda_engine(f2.read())
context2 = engine2.create_execution_context()

# output2fff = np.empty([BATCH_SIZE,*output_shape,128], dtype = target_dtype)
# output2bm = np.empty([BATCH_SIZE,*output_shape,15], dtype = target_dtype)
# output2paf = np.empty([BATCH_SIZE,*output_shape,32], dtype = target_dtype)
output2 = np.empty([BATCH_SIZE,*output_shape,175], dtype = target_dtype)

d_input2 = cuda.mem_alloc(1 * dummy_input_batch.nbytes)

# d_output2fff = cuda.mem_alloc(1 * output2fff.nbytes)
# d_output2bm = cuda.mem_alloc(1 * output2bm.nbytes)
# d_output2paf = cuda.mem_alloc(1 * output2paf.nbytes)
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

# output3tff = np.empty([BATCH_SIZE,*output_shape,30], dtype = target_dtype)
# output3bm0 = np.empty([BATCH_SIZE,*output_shape,15], dtype = target_dtype)
# output3bm1 = np.empty([BATCH_SIZE,*output_shape,15], dtype = target_dtype)
# output3paf0 = np.empty([BATCH_SIZE,*output_shape,32], dtype = target_dtype)
# output3paf1 = np.empty([BATCH_SIZE,*output_shape,32], dtype = target_dtype)
output3 = np.empty([BATCH_SIZE,*output_shape,124], dtype = target_dtype)

d_input3 = cuda.mem_alloc(1 * dummy_input_batch0.nbytes)

# d_output3tff = cuda.mem_alloc(1 * output3tff.nbytes)
# d_output3bm0 = cuda.mem_alloc(1 * output3bm0.nbytes)
# d_output3bm1 = cuda.mem_alloc(1 * output3bm1.nbytes)
# d_output3paf0 = cuda.mem_alloc(1 * output3paf0.nbytes)
# d_output3paf1 = cuda.mem_alloc(1 * output3paf1.nbytes)
d_output3 = cuda.mem_alloc(1 * output3.nbytes)

bindings3 = [int(d_input3),int(d_output3)]
stream3 = cuda.Stream()

print('Generating TFF: Done')

print('Warming up ...') # To modify
prediction0 = predict_open(dummy_input_batch,context2,bindings2,d_input2,d_output2,stream2,output2)
prediction = predict_TFF(dummy_input_batch0,context3,bindings3,d_input3,d_output3,stream3,output3)
print('Done Warming up!')

print('Processing frames: ...')

Q,processing_times = joint_flow_from_dir(a.input_dir,input_shape,output_shape,context2,bindings2,d_input2,d_output2,stream2,output2,context3,bindings3,d_input3,d_output3,stream3,output3)

print('Processing frames: done')

print('Creating video: ...')
# clear_output_folder('.')
make_vid_from_dict_joint_flow(Q,processing_times,a.input_dir,'JF_'+argument_list[2]+'_'+argument_list[3]+'_output.mp4') 
print('Creating video: done')


print('Creating JSON: ...')
write_processing_times_JSON_JF(processing_times,'JF_'+argument_list[2]+'_'+argument_list[3]+'_processing_time.json')
write_Q_JSON_JF(Q,'JF_'+argument_list[2]+'_'+argument_list[3]+'_Q.json')
print('Creating JSON: done')