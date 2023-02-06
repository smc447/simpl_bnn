import heterocl as hcl
import hlib.bnn as bnn
import numpy as np
import checks as c
import inputs as i

hcl.init()
def bnn_dense(dense_input, w):
    dense1 = bnn.dense(dense_input,w, None ,True, dtype = hcl.UInt(1) )
    return dense1

dense_input = hcl.placeholder((1,512),"input", hcl.UInt(1))
w  = hcl.placeholder((256,512), "w", hcl.UInt(1))
s = hcl.create_schedule([INPUT,w], bnn_dense)
f = hcl.build(s)

np_input = c.np_flatten
hcl_input = hcl.asarray(np_input, dtype = hcl.UInt(1))
np_wfc1 = wfc.np1_wfc1.reshape(256,512)
hcl_w = hcl.asarray(np_wfc1, hcl.UInt(1))
np_output = np.zeros((1,512))
hcl_output = hcl.asarray(np_output, dtype = hcl.UInt(1))

f(hcl_input, hcl_output)
out = hcl_output.asnumpy()
print(out.shape)
print(c.np_flatten.shape)
assert np.array_equal(out, c.np_flatten)
print("done")