import heterocl as hcl
import heterocl.op.bnn as bnn
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
s[bnn_conv.conv1].reorder( bnn_conv.conv1.axis[3],  bnn_conv.conv1.axis[2] ) #x is axis 3 and y is axis 2


#hcl.reorder(%s, %lrc, %lry, %lrx, %lj)
#%buf = hcl.buffer_at(%s, %Output: memref<6x30x30xf32>, %li) -> memref<30xf32>

s[bnn_conv.conv1].reorder( bnn_conv.conv1.axis[4],  bnn_conv.conv1.axis[5], bnn_conv.conv1.axis[2] )
s.buffer_at(bnn_conv.conv1, s[bnn_conv.conv1], bnn_conv.conv1.axis[3])


#Other optimization attempts:
#s[bnn_conv.conv1].reorder( bnn_conv.conv1.axis[4],  bnn_conv.conv1.axis[5], bnn_conv.conv1.axis[3] )
#s.buffer_at(bnn_conv.conv1, s[bnn_conv.conv1], bnn_conv.conv1.axis[2])

#[bnn_conv.conv1].reorder( bnn_conv.conv1.axis[4],  bnn_conv.conv1.axis[5], bnn_conv.conv1.axis[2], bnn_conv.conv1.axis[3] )
#s.buffer_at(bnn_conv.conv1, s[bnn_conv.conv1], bnn_conv.conv1.axis[2])

# s[bnn_conv.conv1].reorder( bnn_conv.conv1.axis[4],  bnn_conv.conv1.axis[5], bnn_conv.conv1.axis[3], bnn_conv.conv1.axis[2] )
# s.buffer_at(bnn_conv.conv1, s[bnn_conv.conv1], bnn_conv.conv1.axis[3])

# s[bnn_conv.conv1].reorder( bnn_conv.conv1.axis[0], bnn_conv.conv1.axis[1], bnn_conv.conv1.axis[4],  bnn_conv.conv1.axis[5], bnn_conv.conv1.axis[2] )
# s.buffer_at(bnn_conv.conv1, s[bnn_conv.conv1], bnn_conv.conv1.axis[3])

print(hcl.lower(s))

f_sim = hcl.build(s, target="vhls")
print(f_sim)