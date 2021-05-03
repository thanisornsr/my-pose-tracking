from yolo_classes import get_cls_dict
from yolo_with_plugins import get_input_shape, TrtYOLO

import cv2
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

h = 416
w = 416
img_dir = './input/000000.jpg'

#initiate Yolo
conf_th = 0.9
yolo_model = 'yolov4-416'
trt_yolo = TrtYOLO(yolo_model, (h, w))

#initiate Pose_es
BATCH_SIZE = 1
dummy_input_batch = np.zeros((BATCH_SIZE,256,192,3),dtype = np.float32)
USE_FP16 = False
target_dtype = np.float16 if USE_FP16 else np.float32
f2 = open("resnet_engine.trt","rb")
runtime2 = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine2 = runtime2.deserialize_cuda_engine(f2.read())
context2 = engine2.create_execution_context()
output2 = np.empty([BATCH_SIZE,64,48,15], dtype = target_dtype)
d_input2 = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
d_output2 = cuda.mem_alloc(1 * output2.nbytes)
bindings2 = [int(d_input2),int(d_output2)]
stream2 = cuda.Stream()

def predict(batch):
    cuda.memcpy_htod_async(d_input2,batch,stream2)

    context2.execute_async_v2(bindings2,stream2.handle,None)

    cuda.memcpy_dtoh_async(output2, d_output2, stream2)

    stream2.synchronize()

    return output2



img = cv2.imread(img_dir)
print(img.dtype)
ori_shape = img.shape

print('Warming up ...')
predict(dummy_input_batch)
boxes, confs, clss = trt_yolo.detect(img,conf_th)
print('Done Warming up!')


boxes, confs, clss = trt_yolo.detect(img,conf_th)
print(type(boxes))


# print('confs: {}'.format(confs))
# print('clss: {}'.format(clss))
boxes = boxes[clss==0]
print('boxes: {}'.format(boxes))

boxes = boxes.tolist()
for box in boxes:
    print(box)
    print(type(box))

# for i in range(boxes.shape[0]):
#     temp_bbox = boxes[i,:]
#     x0,y0,x1,y1 = temp_bbox
#     crop = img[y0:y1,x0:x1,:]
#     crop = cv2.resize(crop,(192,256))
#     crop2 = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
#     crop2 = np.reshape(crop2,(BATCH_SIZE,256,192,3))
#     cv2.imshow('crop',crop)
#     # print(crop2.dtype)
#     prediction = predict(crop2.astype('float32') / 255.0)
#     # prediction = np.nan_to_num(prediction)
#     # print(crop2)
#     # print(prediction.shape)
#     # print(prediction.dtype)
#     t_max = np.amax(prediction[0,:,:,0])
#     print(t_max)
#     # print(prediction[0,:,:,0])
#     i,j = np.where(prediction[0,:,:,0]==t_max)

#     t_shape = prediction[0,:,:,0].shape
#     om = np.zeros(t_shape)
#     pp = cv2.circle(om,(j[0],i[0]),radius=3,color = 255,thickness=-1)

#     cv2.imshow('pred',pp)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


