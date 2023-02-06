import heterocl as hcl
from scipy import signal
import hlib.bnn as bnn
import numpy as np
import checks as c
import inputs as i
m = 16
h = 16
hcl.init()
def bnn_conv(INPUT, w_conv1):
    conv1 = bnn.conv2d_nchw(INPUT,w_conv1, padding=[1, 1], name="conv1", out_dtype=hcl.Int(6))
    return conv1

INPUT = hcl.placeholder((1,1,h,h),"input", hcl.UInt(1))
w_conv1 = hcl.placeholder((m,1,3,3),"w_conv1", hcl.UInt(1))
s = hcl.create_schedule([INPUT, w_conv1], bnn_conv)
f = hcl.build(s)
#np_w1 = i.np_w1.reshape((16,1,3,3))
#hcl_w1 = hcl.asarray(np_w1, dtype= hcl.UInt(1))
#np_image = i.np_image.reshape((1,1,16,16))
#hcl_image = hcl.asarray(np_image, dtype = hcl.UInt(1))
#np_output = np.zeros((1,16,16,16))
#hcl_output = hcl.asarray(np_output, dtype = hcl.Int(6))
#
#f(hcl_image, hcl_w1, hcl_output)
#out = hcl_output.asnumpy()
#out = out.flatten()
#assert np.array_equal(out, c.conv_only)
#print("done")


def if_mac(x, y, I):
    if (x < 1 or x >= (I - 1) or y < 1 or y >= (I - 1)):
            return False
    return True
def padd(input, M, I):
    outp= np.zeros((I+2, I+2))
    for y in range(I):
        for x in range(I):
            outp[y+1, x+1] = input[y, x]
    return outp
    

def bin_conv(input_test, w_test, N, H, W, M):
    out = np.empty((1,m,h,h))
    image0 = input_test.reshape((h,h))
    image = padd(image0, 1, h)
    w_f_all = w_test.reshape((N,3,3))
    for n in range(N):
        w_f = w_f_all[n].reshape((3,3))
        for y in range(H):
            for x in range(W):
                sum_c = 0
                for mi in range(M):
                    mac_num = 0
                    one_out = 0
                    for c in range(3):
                        for r in range(3):
                            if if_mac(x+c, y+r, h+2):
                                one_out += image[y+r, x+c]==w_f[r,c]
                                mac_num += 1
                    sum_c += (one_out << 1) - mac_num
                out[0,n,y,x] = sum_c
    return out.astype(int)


input_test = np.random.randint(0,2,(1,1,h,h))
w_test = np.random.randint(0,2,(m,1,3,3))
out_check = bin_conv(input_test,w_test,m, h, h, 1)
print(out_check)
#out_check = out_check.flatten()
print(out_check.shape)
hcl_image = hcl.asarray(input_test, dtype = hcl.UInt(1))
hcl_w1 = hcl.asarray(w_test, dtype= hcl.UInt(1))
np_output2 = np.zeros((1,m,h,h))
hcl_output2 = hcl.asarray(np_output2, dtype = hcl.Int(6))

f(hcl_image, hcl_w1, hcl_output2)
out2 = hcl_output2.asnumpy()
print(out2)
#out2 = out2.flatten()
print(out2.shape)

assert np.array_equal(out2, out_check )


