import math
from numba import vectorize, cuda
import numpy as np
import pdb

@vectorize(['float32(float32, float32, float32)',
            'float64(float64, float64, float64)'],
           target='cuda')
def cu_discriminant(a, b, c):
    return math.sqrt(b ** 2 - 4 * a * c)

N = 100000000
dtype = np.float32

# prepare the input
A = np.array(np.random.sample(N), dtype=np.float32)
B = np.array(np.random.sample(N) + 10.0, dtype=np.float32)
C = np.array(np.random.sample(N), dtype=np.float32)
#pdb.set_trace()
D = cu_discriminant(A, B, C)

print(D)  # print result