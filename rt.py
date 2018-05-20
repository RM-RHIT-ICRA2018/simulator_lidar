
from numba import cuda
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@cuda.jit(debug=True)
def cufftShift_2D_kernel(data, N):
    #adopting cuda shift code from:
    #https://github.com/marwan-abdellah/cufftShift
    #GNU Lesser public license
    
    #// 2D Slice & 1D Line
    sLine = N
    sSlice = N * N
    #// Transformations Equations
    sEq1 = int((sSlice + sLine) / 2)
    sEq2 = int((sSlice - sLine) / 2)
    x, y = cuda.grid(2)
    #// Thread Index Converted into 1D Index
    index = (y * N) + x
    #T regTemp;
    #data[index]=0
    if (x < N / 2):
        if (y < N / 2):
            #// First Quad
            temp =data[index]
            data[index] = data[index + sEq1]
            #// Third Quad
            data[index + sEq1] = temp
    else:
        if (y < N / 2):
            #// Second Quad
            temp=data[index]
            data[index] = data[index + sEq2];
            data[index + sEq2] = temp

    
n=4
array=np.ones([n,n])#,dtype=np.complex128)

cufftShift_2D_kernel(array.ravel(),n)