import heterocl as hcl
from scipy import signal
import hlib.bnn as bnn
import numpy as np

#m = 8
#h = 8
#n = 1
#ch = 8
input_size= (1,8,8,8)
w_size = (16,8,3,3)
output_size = (1,16,8,8)
hcl.init()
def bnn_conv(INPUT, w_conv1):
    conv1 = bnn.conv2d_nchw(INPUT,w_conv1, padding=[1, 1], name="conv1", out_dtype=hcl.Int(6))
    return conv1

INPUT = hcl.placeholder(input_size,"input", hcl.Int(2))
w_conv1 = hcl.placeholder(w_size,"w_conv1", hcl.Int(2))
s = hcl.create_schedule([INPUT, w_conv1], bnn_conv)
f = hcl.build(s)

def if_mac(x, y, I):
    if (x < 1 or x >= (I - 1) or y < 1 or y >= (I - 1)):
            return False
    return True
def padd(inputs, I):
    input_shape = inputs.shape
    outp= np.zeros((1,input_shape[1],I+2, I+2))
    for m in range(input_shape[1]):
        for y in range(I):
            for x in range(I):
                outp[0, m, y+1, x+1] = inputs[0,m,y, x]
    return outp
    

def bin_conv(input_test, w_test, w_shape, input_shape):
    out = np.empty((1,w_shape[0],input_shape[2], input_shape[3]))
    image0 = padd(input_test, input_shape[2])
    for ni in range(w_shape[0]):
        for y in range(input_shape[2]):
            for x in range(input_shape[3]):
                sum_c = 0
                for mi in range(input_shape[1]):
                    image = image0[0,mi]
                    w_f = w_test[ni, mi]
                    mac_num = 0
                    one_out = 0
                    for c in range(3):
                        for r in range(3):
                            if if_mac(x+c, y+r, input_shape[2]+2):
                                one_out += image[y+r, x+c]==w_f[r,c]
                                mac_num += 1
                    sum_c += (one_out << 1) - mac_num
                out[0,ni,y,x] = sum_c
    return out.astype(int)

def py_conv(input_test, w_test):
    w_shape = w_test.shape
    input_shape = input_test.shape
    image = padd(input_test)



input_test = np.random.randint(0,2,input_size)
input_test_trans = np.where(input_test ==0, -1, 1)
w_test = np.random.randint(0,2,w_size)
w_test_trans = np.where(w_test ==0, -1, 1)
out_check = bin_conv(input_test,w_test, w_size, input_size)
#print(out_check)
#out_check = out_check.flatten()
print(out_check.shape)
hcl_image = hcl.asarray(input_test_trans, dtype = hcl.Int(2))
hcl_w1 = hcl.asarray(w_test_trans, dtype= hcl.Int(2))
np_output2 = np.zeros(output_size)
hcl_output2 = hcl.asarray(np_output2, dtype = hcl.Int(6))

f(hcl_image, hcl_w1, hcl_output2)
out2 = hcl_output2.asnumpy()
#print(out2)
#out2 = out2.flatten()
print(out2.shape)

assert np.array_equal(out2, out_check )


