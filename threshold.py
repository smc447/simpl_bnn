
import numpy as np
np1_threshold1 = np.array([1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,1.74385,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.273957,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,-0.494261,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,5.20923,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-2.91049,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-0.959595,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-1.73834,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-3.24419,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,-0.870229,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,2.12276,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,3.68238,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,1.2267,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,4.72314,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-0.231407,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-2.28068,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729,-1.17729])