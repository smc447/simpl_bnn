import heterocl as hcl
import hlib.bnn as bnn
import numpy as np
import checks as ch
import inputs as i
n = 1
c = 16
h = 8
w = 8
hcl.init()
def bnn_pool(conv1_thresh):
    conv1_pool = bnn.max_pool2d_nchw(conv1_thresh,[2, 2], [2, 2], name="conv1_pool" )
    return conv1_pool

conv1_thresh = hcl.placeholder((n,c,h*2,w*2),"input", hcl.UInt(1))
s = hcl.create_schedule([conv1_thresh], bnn_pool)
f = hcl.build(s)

np_thresh = ch.np_first_conv.reshape((1,16,16,16))
hcl_thresh = hcl.asarray(np_thresh, dtype = hcl.UInt(1))
np_output = np.zeros((1,16,8,8))
hcl_output = hcl.asarray(np_output, dtype = hcl.UInt(1))

f(hcl_thresh, hcl_output)
out = hcl_output.asnumpy()
out = out.flatten()
assert np.array_equal(out, ch.np_conv_layer1)


def mpool(input_test, n, c, h, w):
    out = np.empty((n,c,h,w))
    for ni in range(n):
        for ci in range(c):
            for y in range(h):
                for x in range(w):
                    m_sum = 0
                    for i in range(2):
                        for j in range(2):
                            if(input_test[ni,ci, 2*y+ i, 2*x+j]):
                                m_sum = 1
                    out[ni,ci,y,x] = m_sum
    return out
    

input_test = np.random.randint(0,2,(n,c,h*2,w*2))
result = mpool(input_test,n,c,h,w)

hcl_thresh = hcl.asarray(input_test, dtype = hcl.UInt(1))
np_output2 = np.zeros((n,c,h,w))
hcl_output2 = hcl.asarray(np_output2, dtype = hcl.UInt(1))

f(hcl_thresh, hcl_output2)
out2 = hcl_output2.asnumpy()

assert np.array_equal(out2, result)


print("done")