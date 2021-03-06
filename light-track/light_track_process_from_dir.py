import argparse
from light_track_utils import *
import glob
import numpy as np
from yolo_classes import get_cls_dict
from yolo_with_plugins import get_input_shape, TrtYOLO

import cv2
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def read_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('input_dir')

	args = parser.parse_args()

	return args 

a = read_arguments()

print('Input folder directory: {}'.format(a.input_dir))


# config 
h = 416
w = 416
conf_th = 0.95
yolo_model = 'yolov4-416'
BATCH_SIZE = 1
USE_FP16 = False # USE FP32
input_shape = (384, 288)
output_shape = (96,72)
KEY_FRAME = 2
argument_list = a.input_dir.split('/')

# build human detector
print('Generating human detector: ...')

trt_yolo = TrtYOLO(yolo_model, (h, w))

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

temp_file_list = sorted(glob.glob(a.input_dir+'/*.jpg'))
img = cv2.imread(temp_file_list[0])
predict(dummy_input_batch,context2,bindings2,d_input2,d_output2,stream2,output2)
boxes, confs, clss = trt_yolo.detect(img,conf_th)

print('Done Warming up!')

# run flow_track_from_dir
print('Processing frames: ...')

Q,processing_times = light_track_from_dir(a.input_dir,trt_yolo,context2,bindings2,d_input2,d_output2,stream2,output2,input_shape,output_shape,KEY_FRAME)

print('Processing frames: done')

print('Creating video: ...')
# clear_output_folder('.')
make_video_light_track(a.input_dir,processing_times,Q,'LT_'+argument_list[2]+'_'+argument_list[3]+'_output.mp4') 
print('Creating video: done')

print('Creating JSON: ...')
write_processing_times_JSON(processing_times,'LT_'+argument_list[2]+'_'+argument_list[3]+'_processing_time.json')
write_Q_JSON(Q,'LT_'+argument_list[2]+'_'+argument_list[3]+'_Q.json')
print('Creating JSON: done')
