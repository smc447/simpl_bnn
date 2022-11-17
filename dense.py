import heterocl as hcl
import hlib.bnn as bnn
import hlib.nn as nn
import numpy as np
import wfc as wfc
def dense(conv, w, b):
    dense_input = bnn.flatten(conv, name = "flatten")
    print(dense_input.shape)
    dense1 = bnn.dense(dense_input,w, None ,True, dtype = hcl.UInt(1) )
    return dense1

conv= hcl.placeholder((1, 32, 4,4),"input", hcl.UInt(1))
w = hcl.placeholder((256,512), "w", hcl.UInt(1))
b = hcl.placeholder((256,), "b", hcl.Float(64,32))
s = hcl.create_schedule([conv, w, b], dense)
f = hcl.build(s)

np_conv = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
np_conv = np_conv.reshape((1,32,4,4))
hcl_conv = hcl.asarray(np_conv,  dtype = hcl.UInt(1))

#np_reshape = np.zeros((1, 512))
#hcl_reshape = hcl.asarray(np_reshape,  dtype = hcl.UInt(1))


np_wfc1 = wfc.np1_wfc1.reshape(256,512)
hcl_w = hcl.asarray(np_wfc1, hcl.UInt(1))

np_bfc1 = np.array([0.0061140214,-0.0035128274,0.0078034941,-0.0027952984,0.0064750859,0.0111233015,-0.0030463664,0.0131766712,-0.0092927096,0.0157721601,0.0069579151,-0.0143890865,0.0204967074,-0.0018101599,0.0011093523,0.0046689422,-0.0144519648,0.0018972557,-0.0031878371,0.0077594034,-0.0096419472,-0.0010297339,0.0096812006,-0.0072366553,0.0116966879,0.0116591267,0.0146859670,0.0072759204,-0.0111711379,-0.0088897236,-0.0002260160,0.0005925030,-0.0071704304,0.0037323807,0.0078574186,-0.0044409088,-0.0059927595,-0.0014872052,0.0013591378,0.0054184832,0.0105621340,-0.0047257091,0.0147370873,0.0014799575,-0.0119214198,-0.0047542998,-0.0021814085,0.0046050465,0.0015757388,0.0118793715,-0.0047356049,0.0043867650,0.0025899587,0.0060640899,0.0005729974,0.0111787105,-0.0060605635,-0.0052803592,0.0040420787,0.0086319577,-0.0042159669,0.0079145925,-0.0023865597,0.0021521468,-0.0164503269,-0.0083829034,-0.0038342427,-0.0033420972,0.0133293215,0.0014267839,-0.0129346419,-0.0157589950,-0.0001402088,-0.0137956878,0.0079953149,-0.0131986225,0.0190454833,0.0129790325,0.0002455539,-0.0044806651,-0.0026770488,-0.0047985609,-0.0053475029,-0.0040724273,0.0004667633,-0.0055322358,-0.0134633677,0.0136837699,0.0094585624,0.0029087639,0.0096289488,0.0124773141,-0.0088242516,0.0013872589,0.0031195031,-0.0021356621,0.0003840631,-0.0139491428,0.0009918235,0.0165250786,-0.0051826700,-0.0156550854,-0.0027364308,-0.0046837991,0.0019061846,-0.0006821529,-0.0006982983,-0.0117991082,0.0002344730,0.0025774916,0.0018998729,-0.0000028316,-0.0065811784,-0.0067946343,0.0021541175,0.0044194786,0.0045414940,0.0013413373,0.0067321751,0.0151742809,-0.0019494446,-0.0074260216,0.0012232425,-0.0064125704,0.0003995456,0.0079342751,0.0072016898,-0.0107705398,-0.0082389461,0.0010324605,-0.0147075830,0.0030663316,0.0053717974,-0.0035150845,0.0020252231,-0.0015040513,0.0023008254,-0.0049660453,0.0008548130,0.0159292296,0.0029085991,0.0018672622,-0.0066935853,-0.0095038516,-0.0060322271,0.0064574806,-0.0044140546,0.0090418393,-0.0013561090,0.0102996957,-0.0033241489,-0.0023035749,-0.0056283758,-0.0060749450,0.0039814352,-0.0061721751,0.0083528068,-0.0059141084,0.0055607269,0.0002878043,0.0153128039,-0.0015419924,0.0140372142,0.0028955257,0.0028704796,-0.0013018097,-0.0021185391,0.0099513428,0.0003111081,-0.0014089026,0.0135370092,0.0039698039,-0.0084944321,0.0077125472,-0.0039506899,0.0037965798,-0.0028132901,0.0161214899,-0.0066210842,-0.0098690363,-0.0143555338,-0.0130735468,0.0042721475,0.0110243540,-0.0031558529,-0.0074020401,0.0115068071,-0.0068032178,-0.0077676098,0.0015251224,0.0121638132,0.0015165495,-0.0013726463,-0.0101275947,0.0043065636,0.0063501666,0.0041412683,0.0065069324,0.0093019530,-0.0011172802,0.0002398869,0.0011896160,0.0038999093,-0.0013471507,-0.0091317752,-0.0017160014,0.0045599435,0.0240042694,0.0029732892,0.0009912843,0.0048813443,-0.0092924032,0.0132083949,0.0071377452,-0.0008864424,0.0020607389,0.0005941214,0.0143124294,-0.0032886697,0.0016461053,0.0050705103,-0.0085436786,-0.0000544414,0.0011244335,0.0084771290,0.0050083515,0.0042721462,-0.0094863726,0.0029722403,-0.0016605321,0.0017104452,-0.0013858913,-0.0059214472,0.0045991009,-0.0030122988,0.0043640835,-0.0047134194,0.0058397059,0.0075935614,-0.0052441363,-0.0006699506,0.0004899215,-0.0038960094,0.0028124452,0.0075199949,-0.0097959377,0.0109991273,0.0003967878,0.0013354269,-0.0038244906,0.0041599954,-0.0009608836,-0.0156793408,-0.0049158968,-0.0019097127,0.0149538880
])

hcl_b = hcl.asarray(np_bfc1, dtype = hcl.Float(64,32))

np_dout = np.zeros(1,256)
hcl_dout = hcl.asarray(np_dout, hcl.UInt(1))

f(hcl_conv, hcl_w, hcl_b, hcl_dout)
out = hcl_reshape.asnumpy()
out = out.flatten()
expected_reshape = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1])
print(out)

assert np.array_equal(expected, out)