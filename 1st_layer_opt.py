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
s_conv = bnn_conv_thresh_pool.conv1
s_thresh = bnn_conv_thresh_pool.conv1_thresh
f_sim = hcl.build(s, target="vhls")
print(f_sim)

print("done")
#thresh = t.np1_threshold1.reshape((16,16,16))
##print(thresh)
#for i in range(16):
#    print("{", end=" ")
#    for j in range(16):
#        print("{",end=" ")
#        for n in range(16):
#            if n == 15:
#                print(thresh[i][j][n], end=" ")
#            else:
#                print(thresh[i][j][n], end=", ")
#        if j == 15:
#            print("}", end = " ")
#        else:
#            print("}", end=", ")
#    print("}", end=", ")