import numpy as np 

BATCH_SIZE = 1
dummy_input_batch = np.ones((BATCH_SIZE,256,192,3))

PRECISION = 'FP32'

from helper import ModelOptimizer,OptimizedModel

model_dir = './model/flow_track_r152'
optimized_model_dir = './model/flow_track_r152_FP32'

# opt_model = ModelOptimizer(model_dir)

# model_fp32 = opt_model.convert(model_dir+'_FP32',precision=PRECISION)

model_fp32 = OptimizedModel(optimized_model_dir)

