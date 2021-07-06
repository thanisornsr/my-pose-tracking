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

USE_FP16 = False # USE FP32

# build openpose
print('Generating Openpose: ...')

dummy_input_batch = np.zeros((BATCH_SIZE,*input_shape,3),dtype = np.float32)
target_dtype = np.float16 if USE_FP16 else np.float32

f2 = open("lt_resnet152_engine.trt","rb") # change name to open pose
runtime2 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine2 = runtime2.deserialize_cuda_engine(f2.read())
context2 = engine2.create_execution_context()
output2 = np.empty([BATCH_SIZE,*output_shape,15], dtype = target_dtype)
d_input2 = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
d_output2 = cuda.mem_alloc(1 * output2.nbytes)
bindings2 = [int(d_input2),int(d_output2)]
stream2 = cuda.Stream()

print('Generating Openpose: Done')

# build TFF # To modify
# print('Generating TFF: ...')

# dummy_input_batch = np.zeros((BATCH_SIZE,*input_shape,3),dtype = np.float32)
# target_dtype = np.float16 if USE_FP16 else np.float32

# f2 = open("lt_resnet152_engine.trt","rb") # change name to open pose
# runtime2 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
# engine2 = runtime2.deserialize_cuda_engine(f2.read())
# context2 = engine2.create_execution_context()
# output2 = np.empty([BATCH_SIZE,*output_shape,15], dtype = target_dtype)
# d_input2 = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
# d_output2 = cuda.mem_alloc(1 * output2.nbytes)
# bindings2 = [int(d_input2),int(d_output2)]
# stream2 = cuda.Stream()

# print('Generating TFF: Done')

# print('Warming up ...') # To modify

# temp_file_list = sorted(glob.glob(a.input_dir+'/*.jpg'))
# img = cv2.imread(temp_file_list[0])
# predict(dummy_input_batch,context2,bindings2,d_input2,d_output2,stream2,output2)
# boxes, confs, clss = trt_yolo.detect(img,conf_th)

# print('Done Warming up!')