import numpy as np
from numba import jit
from md import pbc

@jit(nopython=True)
def neighbourlist(pos,rs,shape_params,pbc_params,maxnbrs):
    
    nc = np.zeros((shape_params[0]),dtype=np.intc)
    nclist = np.zeros((shape_params[0],maxnbrs),dtype=np.intc)
    r=np.zeros((shape_params[1]),dtype=np.float64) 
    
    #Loop running over all particle pairs
    for i in range(shape_params[0]):
        for j in range(i+1,shape_params[0]):
            
            
            #Minimum image criteria distance computation
            rnorm,r = pbc.dist_mic(shape_params[1],pos[i,:],pos[j,:],pbc_params[0],pbc_params[1])
            
            if (rnorm<rs):
                nc[i]=nc[i]+1
                nclist[i,nc[i]]=j
    
    
    return(nc,nclist)