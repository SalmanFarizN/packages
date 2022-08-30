import numpy as np
from md import pbc

#Function to initialise position of particles so that they do not overlap .. run only once at the beginning 
def posini(shape_params,pbc_params,diam):
    pos1=np.zeros((shape_params[0],shape_params[1]),dtype=np.float64)
    sep=np.zeros((shape_params[0]),dtype=np.float64)
    dimsep=np.zeros((shape_params[1]),dtype=np.float64)
    
    for i in range(shape_params[0]):
        print(i)
        if i == 0:
            for j in range(shape_params[1]):
                pos1[i,j]=np.random.uniform(0.0,pbc_params[0])
        else:
            for j in range(shape_params[1]):
                pos1[i,j]=np.random.uniform(0.0,pbc_params[0])
            
            for k in range(i):
                if i!=k:
                    #Computing distance with the mirror image condition
                    sep[k],r = pbc.dist_mic(shape_params[1],pos1[i,:],pos1[k,:],pbc_params[0],pbc_params[1])
                
            while(any(q < diam and q!=0.0 for q in sep)):
                for j in range(shape_params[1]):
                    pos1[i,j]=np.random.uniform(0.0,pbc_params[0])
                for k in range(i):
                    if i!=k:
                        #Computing distance with the mirror image condition
                        sep[k],r = pbc.dist_mic(shape_params[1],pos1[i,:],pos1[k,:],pbc_params[0],pbc_params[1])
    
    return(pos1)