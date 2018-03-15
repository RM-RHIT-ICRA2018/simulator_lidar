from numba import cuda


@cuda.jit
def add_kernel(data, y, out):
    tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x    # number of blocks in the grid
    angle=data[tx][1]
    if data[tx][2]==0:
        laser_dis=maxdist
    else:
        laser_dis=data[tx][2]
    
    if not(angle==-1):
        pos=particle_pos[ty]
        out[ty][tx]=(get_laser_ref(all_obstacle_segments,angle,pos)-laser_dis)**2

    # start = tx + ty * block_size
    # stride = block_size * grid_size

    # # assuming x and y inputs are same length
    # for i in range(start, x.shape[0], stride):
    #     out[i] = x[i] + y[i]

