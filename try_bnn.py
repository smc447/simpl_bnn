import heterocl as hcl
import hlib.bnn as bnn
import hlib.nn as nn
import numpy as np
def bnn_main(INPUT, w_conv1,  threshold):
    conv1 = bnn.conv2d_nchw(INPUT,w_conv1, padding=[1, 1], name="conv1", out_dtype=hcl.Int(6))
    #conv1_thresh = bnn.batch_norm_threshold(conv1,threshold, name="conv1_thresh")
    #return conv1_thresh
    return conv1

INPUT = hcl.placeholder((1,1,3,3),"input", hcl.UInt(1))
w_conv1 = hcl.placeholder((1,1,3,3),"w_conv1", hcl.UInt(1))
#threshold = hcl.placeholder((1,3,3), "threshold", hcl.Fixed(32, 16))
threshold = hcl.placeholder((1,3,3), "threshold", hcl.Int(8))
s = hcl.create_schedule([INPUT, w_conv1, threshold], bnn_main)
f = hcl.build(s)
np_w1 = np.array([1 ,0,0,1 ,0,0,0,1 ,1])
np_w1 = np_w1.reshape((1,1,3,3))
hcl_w1 = hcl.asarray(np_w1, dtype= hcl.UInt(1))
np_image = np.array([0,0,0,1,0,0,0,1,0])
np_image = np_image.reshape((1,1,3,3))
hcl_image = hcl.asarray(np_image, dtype = hcl.UInt(1))
np_threshold = np.array([1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385])
np_threshold = np_threshold.reshape(1,3,3)
#hcl_threshold = hcl.asarray(np_threshold,  dtype = hcl.Fixed(32, 16))
hcl_threshold = hcl.asarray(np_threshold,  dtype = hcl.Int(8))
np_pool = np.zeros((1,1,3,3))
hcl_conv1_pool = hcl.asarray(np_pool, dtype = hcl.Int(6))
#hcl_conv1_pool = hcl.asarray(np_pool, dtype = hcl.UInt(1))
f(hcl_image,hcl_w1, hcl_threshold, hcl_conv1_pool)
out = hcl_conv1_pool.asnumpy()
out = out.flatten()
np_check = np.array([2, -2, 0, 0, 5, 2, 2, 2, -2])
print(out)
