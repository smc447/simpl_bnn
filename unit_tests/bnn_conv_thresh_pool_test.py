import heterocl as hcl
import hlib.bnn as bnn
import numpy as np
import checks as c
import inputs as i
import threshold as t

hcl.init()
def bnn_conv_thresh_pool(INPUT, w_conv1, t1):
    conv1 = bnn.conv2d_nchw(INPUT,w_conv1, padding=[1, 1], name="conv1", out_dtype=hcl.Int(6))
    conv1_thresh = bnn.batch_norm_threshold(conv1, t1, name="conv1_thresh")
    conv1_pool = bnn.max_pool2d_nchw(conv1_thresh,[2, 2], [2, 2], name="conv1_pool" )
    return conv1_pool

INPUT = hcl.placeholder((1,1,16,16),"input", hcl.UInt(1))
w_conv1 = hcl.placeholder((16,1,3,3),"w_conv1", hcl.UInt(1))
t1 = hcl.placeholder((16,16,16), "threshold", hcl.Int(8))
s = hcl.create_schedule([INPUT, w_conv1, t1], bnn_conv_thresh_pool)
f = hcl.build(s)

np_w1 = i.np_w1.reshape((16,1,3,3))
hcl_w1 = hcl.asarray(np_w1, dtype= hcl.UInt(1))
np_t = t.np1_threshold1.reshape((16,16,16))
hcl_t = hcl.asarray(np_t,  dtype = hcl.Int(8))
np_image = i.np_image.reshape((1,1,16,16))
hcl_image = hcl.asarray(np_image, dtype = hcl.UInt(1))
np_output = np.zeros((1,16,8,8))
hcl_output = hcl.asarray(np_output, dtype = hcl.UInt(1))

f(hcl_image, hcl_w1, hcl_t, hcl_output)
out = hcl_output.asnumpy()
out_check = c.np_conv_layer1.reshape((1,16,8,8))
assert np.array_equal(out, out_check)
print("done")