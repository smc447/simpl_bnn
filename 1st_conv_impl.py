import heterocl as hcl
import hlib.bnn as bnn
import numpy as np
import checks as c
import inputs as i

hcl.init()
def bnn_conv(INPUT, w_conv1):
    conv1 = bnn.conv2d_nchw(INPUT,w_conv1, padding=[1, 1], name="conv1", out_dtype=hcl.Int(6))
    return conv1

INPUT = hcl.placeholder((1,1,16,16),"input", hcl.UInt(1))
w_conv1 = hcl.placeholder((16,1,3,3),"w_conv1", hcl.UInt(1))
s = hcl.create_schedule([INPUT, w_conv1], bnn_conv)
f_sim = hcl.build(s, target="vhls")
print(f_sim)

np_w1 = i.np_w1.reshape((16,1,3,3))
hcl_w1 = hcl.asarray(np_w1, dtype= hcl.UInt(1))
np_image = i.np_image.reshape((1,1,16,16))
hcl_image = hcl.asarray(np_image, dtype = hcl.UInt(1))
np_output = np.zeros((1,16,16,16))
hcl_output = hcl.asarray(np_output, dtype = hcl.Int(6))


#for i in range(16):
#    print("{{", end=" ")
#    for j in range(3):
#        print("{",end=" ")
#        for n in range(3):
#            if n == 2:
#                print(np_w1[i][0][j][n], end=" ")
#            else:
#                print(np_w1[i][0][j][n], end=", ")
#        if j == 2:
#            print("}", end = " ")
#        else:
#            print("}", end=", ")
#    print("}}", end=", ")

