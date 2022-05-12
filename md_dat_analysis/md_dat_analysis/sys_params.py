import numpy as np
from numba import jit,objmode
from md import pbc 


@jit(nopython=True)
def rdf(pos,length,rho,rmax,bins,dr,t_interval):
    hlength=length/2.0
    r=np.linspace(0,rmax,bins)
    gr=np.zeros((len(r)),dtype= np.float32)
    ndims=pos.shape[1]
    nparticles=pos.shape[0]
    spoints=pos.shape[2]
    
    for i in range(len(r)):
        width=r[i]+dr
        npp=0
        d=np.zeros((ndims))
        pointden=(nparticles)/(length**ndims) 
        tavg=spoints/t_interval
        
        #Averaging over multiple timesteps
        for l in range(0,spoints+t_interval,t_interval):
            
            #Averaging over all particles
            for j in range(nparticles):
                for k in range(nparticles):
                    
                    if (j!=k):
                        dnorm,d=pbc.dist_mic(ndims,pos[j,:,l],pos[k,:,l],length,hlength)
                        
                        if dnorm>r[i] and dnorm < width:
                            npp=npp+1
                            
            
        avglocalden=npp/(nparticles*tavg)
        avglocalden=(avglocalden)/(3.14*((r[i]+dr)**2-r[i]**2))            #2D
        gr[i]=avglocalden/(pointden)
       
    return(gr,r)