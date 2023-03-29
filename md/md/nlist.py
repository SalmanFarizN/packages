import numpy as np
from numba import jit,prange
from md import pbc


#Function optimized for single core performance
@jit(nopython=True)
def neighbourlist(pos,rs,shape_params,pbc_params,maxnbrs):
    
    nc = np.zeros((shape_params[0]),dtype=np.intc)
    nclist = np.zeros((shape_params[0],maxnbrs),dtype=np.intc)
    r=np.zeros((shape_params[1]),dtype=np.float64) 
    
    #Loop running over all particle pairs
    for i in range(shape_params[0]):
        for j in range(i+1,shape_params[0]):           #Avoid repeatitions in list; for eg: if 0 has 38 in nlist, 38 will not have 0 in its own nlist
            
            
            #Minimum image criteria distance computation
            rnorm,r = pbc.dist_mic(shape_params[1],pos[i,:],pos[j,:],pbc_params[0],pbc_params[1])
            
            if (rnorm<rs):
                nc[i]=nc[i]+1
                nclist[i,nc[i]]=j
    
    
    return(nc,nclist)



#Function optimized for multi-core performance
@jit(nopython=True,parallel=True)
def neighbourlist_p(pos,rs,shape_params,pbc_params,maxnbrs):
    
    nc = np.zeros((shape_params[0]),dtype=np.intc)
    nclist = np.zeros((shape_params[0],maxnbrs),dtype=np.intc)
    r=np.zeros((shape_params[1]),dtype=np.float64) 
    
    #Loop running over all particle pairs
    for i in prange(shape_params[0]):        #Loop to be parallelized; 
        for j in range(shape_params[0]):    #Should loop through all particles
            
            if (i!=j):                      #Avoid inclusion of itself in neighbour list
            
                #Minimum image criteria distance computation
                rnorm,r = pbc.dist_mic(shape_params[1],pos[i,:],pos[j,:],pbc_params[0],pbc_params[1])
            
                if (rnorm<rs):
                    nc[i]=nc[i]+1
                    nclist[i,nc[i]]=j
    
    
    return(nc,nclist)