import heterocl as hcl
import heterocl.op.bnn as bnn
import numpy as np
import checks as c
import inputs as i
import threshold as t

hcl.init()
def bnn_conv_thresh_pool(INPUT, w_conv1, t1, w_conv2,t2):
    conv1 = bnn.conv2d_nchw(INPUT,w_conv1, padding=[1, 1], name="conv1", out_dtype=hcl.Int(6))
    conv1_thresh = bnn.batch_norm_threshold(conv1, t1, name="conv1_thresh")
    conv1_pool = bnn.max_pool2d_nchw(conv1_thresh,[2, 2], [2, 2], name="conv1_pool" )
    conv2 = bnn.conv2d_nchw(conv1_pool,w_conv2, padding=[1, 1], name="conv2", out_dtype=hcl.Float())
    conv2_thresh = bnn.batch_norm_threshold(conv2,t2, name="conv2_thresh")
    conv2_pool = bnn.max_pool2d_nchw(conv2_thresh,[2, 2], [2, 2], name="conv2_pool" )
    flatten_conv2 = bnn.flatten(conv2_pool, name = "flatten_conv2")
    return flatten_conv2

INPUT = hcl.placeholder((1,1,16,16),"input", hcl.UInt(1))
w_conv1 = hcl.placeholder((16,1,3,3),"w_conv1", hcl.UInt(1))
t1 = hcl.placeholder((16,16,16), "threshold", hcl.Int(8))
w_conv2 = hcl.placeholder((32,16,3,3), "w_conv2", hcl.UInt(1))
t2 = hcl.placeholder((32,8,8), "threshold2", hcl.Int(8))
s = hcl.create_schedule([INPUT, w_conv1, t1, w_conv2, t2], bnn_conv_thresh_pool)
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