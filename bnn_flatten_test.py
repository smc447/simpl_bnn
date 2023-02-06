import heterocl as hcl
import hlib.bnn as bnn
import numpy as np
import checks as c
import inputs as i

n = 1
m = 32
h = 4
w = 4

hcl.init()
def bnn_flat(INPUT):
    flatten = bnn.flatten(INPUT, name = "flatten")
    return flatten

INPUT = hcl.placeholder((n,m,w,h),"input", hcl.UInt(1))
s = hcl.create_schedule([INPUT], bnn_flat)
f = hcl.build(s)

np_input = c.np_conv_layer2.reshape((1,32,4,4))
hcl_input = hcl.asarray(np_input, dtype = hcl.UInt(1))
np_output = np.zeros((1,512))
hcl_output = hcl.asarray(np_output, dtype = hcl.UInt(1))

f(hcl_input, hcl_output)
out = hcl_output.asnumpy()
print(out.shape)
print(c.np_flatten.shape)
assert np.array_equal(out, c.np_flatten)

def resh(input_test):
    out = np.array([])
    for y in range(h):
        for x in range(w):
            for i in range(m):
                out = np.append(out,input_test[0,i,y,x])
    return out.reshape((1,-1))
input_test = np.random.randint(0,2,(n,m,w,h))
hcl_input = hcl.asarray(input_test, dtype = hcl.UInt(1))
np_output2 = np.zeros((1,m*w*h))
hcl_output2 = hcl.asarray(np_output2, dtype = hcl.UInt(1))
f(hcl_input, hcl_output2)
out2 = hcl_output2.asnumpy()
out_check = resh(input_test)
assert np.array_equal(out2, out_check )
print("done")

