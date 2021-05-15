BATCH_SIZE = 1

import numpy as np
import os
import cv2

USE_FP16 = False

target_dtype = np.float16 if USE_FP16 else np.float32

dummy_input_batch = np.zeros((BATCH_SIZE,256,192,3),dtype = np.float32)


if USE_FP16:
    os.system("/usr/src/tensorrt/bin/trtexec --onnx=lt_resnet152_onnx_model.onnx --saveEngine=lt_resnet152_engine.trt --explicitBatch --fp16")
else:
    os.system("/usr/src/tensorrt/bin/trtexec --onnx=lt_resnet152_onnx_model.onnx --saveEngine=lt_resnet152_engine.trt --explicitBatch")


# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

# f = open("resnet_engine.trt","rb")
# runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

# engine = runtime.deserialize_cuda_engine(f.read())
# context = engine.create_execution_context()

# output = np.empty([BATCH_SIZE,64,48,15], dtype = target_dtype)

# d_input = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
# d_output = cuda.mem_alloc(1 * output.nbytes)

# bindings = [int(d_input),int(d_output)]

# stream = cuda.Stream()

# def predict(batch):
#     cuda.memcpy_htod_async(d_input,batch,stream)

#     context.execute_async_v2(bindings,stream.handle,None)

#     cuda.memcpy_dtoh_async(output, d_output, stream)

#     stream.synchronize()

#     return output

# print('Warming up ...')

# predict(dummy_input_batch)

# print('Done Warming up!')

# img_dir = './test_img/test0.jpg'
# img = cv2.imread(img_dir)
# cv2.imshow('crop',img)
# img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img_batch = np.reshape(img1,(BATCH_SIZE,256,192,3)).astype('float32')
# img_batch = img_batch / 255.0

# prediction = predict(img_batch)
# p = prediction[0,:,:,:]
# print(p[:,:,0])
# print(np.amax(p[:,:,0]))
# i,j = np.where(p[:,:,0]==np.amax(p[:,:,0]))
# print('i,j : {},{}'.format(i,j))

# pp = p[:,:,0]
# pp = cv2.circle(pp,(j[0],i[0]),radius=5,color = 0,thickness=-1)

# cv2.imshow('p',pp)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # print(type(output))
# # print(output.shape)
# # print(output[0,:,:,0])  
# # print(np.amax(output[0,:,:,0]))

