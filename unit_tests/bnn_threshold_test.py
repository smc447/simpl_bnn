import heterocl as hcl
import random
import hlib.bnn as bnn
import numpy as np
import checks as c
import inputs as i
import threshold as t

hcl.init()
def bnn_thresh(INPUT, thresh1):
    t_out = bnn.batch_norm_threshold(INPUT,thresh1, name="t_out")
    return t_out

INPUT = hcl.placeholder((1,16,16,16),"input", hcl.Int(6))
thresh1 =  hcl.placeholder((16,16,16), "threshold", hcl.Int(8))
s = hcl.create_schedule([INPUT, thresh1], bnn_thresh)
f = hcl.build(s)
np_t = t.np1_threshold1.reshape((16,16,16))
hcl_t = hcl.asarray(np_t,  dtype = hcl.Int(8))
np_conv = c.conv_only.reshape((1,16,16,16))
hcl_conv = hcl.asarray(np_conv, dtype = hcl.Int(6))
np_output = np.zeros((1,16,16,16))
hcl_output = hcl.asarray(np_output, dtype = hcl.UInt(1))

f(hcl_conv, hcl_t, hcl_output)
out = hcl_output.asnumpy()
out = out.flatten()
assert np.array_equal(out, c.np_first_conv)


def thresh_test(input_test,t_test):
    return np.where(input_test > t_test, 1, 0)

input_test = np.random.randint(0,32,(1,16,16,16))
t_test = np.random.randint(0,32,(1,16,16,16))
out_check = thresh_test(input_test, t_test)

hcl_t = hcl.asarray(t_test.reshape((16,16,16)),  dtype = hcl.Int(8))
hcl_conv = hcl.asarray(input_test, dtype = hcl.Int(6))
np_output2 = np.zeros((1,16,16,16))
hcl_output2 = hcl.asarray(np_output, dtype = hcl.UInt(1))
f(hcl_conv, hcl_t, hcl_output2)
out2 = hcl_output2.asnumpy()
assert np.array_equal(out2.flatten(), out_check.flatten())
print("done")
