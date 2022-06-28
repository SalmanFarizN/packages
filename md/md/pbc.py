import numpy as np
from numba import jit


# Wall hit check : Entire snapshot array passed in check if any particle is touching the boundary
@jit(nopython=True)
def wallhit(nparticles,ndims,pos,length):
    for i in range(nparticles):
        for j in range(ndims):
            if pos[i,j]< 0.0:
                pos[i,j]+=length
            if pos[i,j]>=length:
                pos[i,j]-=length
    return(pos)





# Wall hit check with unfolding trackers 
# : Entire snapshot array passed in check if any particle is touching the boundary
@jit(nopython=True)
def wallhit_unf(nparticles,ndims,pos,length,tr):
    for i in range(nparticles):
        for j in range(ndims):
            if pos[i,j]< 0.0:
                pos[i,j]+=length
                tr[i,j]-=1
                
            if pos[i,j]>=length:
                pos[i,j]-=length
                tr[i,j]+=1
    return(pos,tr)





# distance computation b/w any 2 particles using the minimum image criteria: pass in co-ords of 2 particles 
@jit(nopython=True)
def dist_mic(ndims,pos1,pos2,length,hlength):
    
    r=np.zeros((ndims),dtype=np.float64)
     
    for k in range(ndims):
        r[k]=pos1[k]-pos2[k]
        if r[k] > hlength:
            r[k]=r[k]-length
        if r[k] <= -hlength:
            r[k]=r[k]+length
            
    rnorm = 0
    for el in range(ndims):
        rnorm = rnorm+r[el]**2
            
    rnorm = np.sqrt(rnorm)
    
    return(rnorm,r)