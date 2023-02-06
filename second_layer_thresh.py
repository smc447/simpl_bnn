import heterocl as hcl
import hlib.bnn as bnn
import hlib.nn as nn
import numpy as np
import threshold as t
import checks as c
import inputs as i
import second_layer_inputs as si

def second_layer_thresh(conv2, threshold2):
    conv2_thresh = bnn.batch_norm_threshold(conv2,threshold2, name="conv1_thresh")
    return conv2_thresh

conv2= hcl.placeholder((1,32,8,8),"input", hcl.Int(6))
threshold2 = hcl.placeholder((32,8,8), "threshold", hcl.Int(8))
s = hcl.create_schedule([conv2, threshold2], second_layer_thresh)
f = hcl.build(s)


np_threshold2_2 = t.np2_threshold2.reshape((32,8,8))
hcl_threshold2 = hcl.asarray(np_threshold2_2,  dtype = hcl.Int(8))
np_conv2 = si.layer2_conv_only.reshape((1,32,8,8))
hcl_conv2 = hcl.asarray(np_conv2, dtype = hcl.Int(6))

np_output = np.zeros((1,32,8,8))
#hcl_output = hcl.asarray(np_output, dtype =hcl.Int(6))
hcl_output = hcl.asarray(np_output, dtype =hcl.UInt(1))

f(hcl_conv2, hcl_threshold2, hcl_output)
out = hcl_output.asnumpy()
#out = out.flatten()
print(si.layer2_conv_only.shape)
print(out.shape)
print(si.layer2_t.shape)
out = out.flatten()
wrong = 0
for i in range(2048):#4096
    if(out[i] != si.layer2_t[i]):
#        print(out[i])
#        print(si.second_layer_thresh[i])
#        print("")
        wrong = wrong +1
print("# of mismatch")
print(wrong)
assert np.array_equal(out, si.layer2_t)
#assert np.array_equal(out, si.second_layer_thresh )



