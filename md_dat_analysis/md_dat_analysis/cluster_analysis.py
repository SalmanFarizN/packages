import numpy as np
from numba import jit,objmode
import cmath
from md import pbc 



#Coordination Number
@jit(nopython=True)
def coord_number(nparticles,ndims,sigma,length,hlength,pos1):
    n_neigh = np.zeros((nparticles),dtype=np.intc)
    neigh_list = np.zeros((nparticles,nparticles),dtype=np.intc)
    r_list = np.zeros((nparticles,nparticles,ndims),dtype=np.float64)
    r = np.zeros((ndims),dtype=np.float64)


    for i in range(nparticles):
        for j in range(nparticles):
            
            if (i!=j):
                rnorm,r=pbc.dist_mic(ndims,pos1[i,:],pos1[j,:],length,hlength)
                
            
                rnorm = 0
                for el in range(ndims):
                    rnorm = rnorm+r[el]**2
            
                rnorm = np.sqrt(rnorm)
                
                if (rnorm<1.15*sigma):
                    n_neigh[i]=n_neigh[i]+1
                    neigh_list[i,n_neigh[i]]=j
                    r_list[i,n_neigh[i],:]=r

            
            
    z_mean=np.sum(n_neigh)/nparticles
    
    return(z_mean,n_neigh,neigh_list,r_list)


#Angular Bond Order parameter
@jit(nopython=True)
def abop(nparticles,ndims,n,r_list,nc):
    phi_avg=0
    count=0
    phi=np.zeros((nparticles),dtype=np.float64)
    
    #x=np.array([[0,0]])
    for i in range(nparticles):
        
        if (nc[i]>1):
            phi[i]=0
            cex=0
            for j in range(nc[i]):
                
                r=r_list[i,j+1,:]
                
                rnorm=0
                for k in range(ndims):
                    rnorm=rnorm+r[k]**2
                rnorm=np.sqrt(rnorm)
                
                if (r[1]>=0):
                    theta=np.arccos(r[0]/rnorm)
                if (r[1]<0):
                    theta=2*np.pi-np.arccos(r[0]/rnorm)
            
                cex=cex+cmath.exp(n*theta*1j)
            
            phi[i]=abs(cex)
            phi[i]=phi[i]/nc[i]
    
            phi_avg=phi_avg+phi[i]
            count=count+1

            
    phi_avg=phi_avg/nparticles
    
    return(phi_avg,phi)
            



#Cluster sizes and lists
@jit(nopython = True)
def cl_sizes(nparticles,ndims,sigma,length,hlength,pos1):
    clusters=np.zeros((nparticles,nparticles),dtype = np.int32)
    clusters[:,:]=-1
    clusters[:,0]=1                   #Size of each cluster
    clusters[:,1]=np.arange(0,1800,1) #First particle in each cluster (1800 clusters initially, each containing 1) 
    r=np.zeros((ndims),dtype=np.float64)


    #Check for each particle
    for i in range(nparticles):
        check=0

        #For each cluster
        for j in range(i):
        
        
            #For each particle inside
            for k in range(clusters[j,0]):
            
                p=clusters[j,k+1]
            
                #Particles are not the same
                if (i!=p):

                    #Compute distance
                    rnorm,r=pbc.dist_mic(ndims,pos1[i,:],pos1[p,:],length,hlength)

                    #If they are closer 
                    if (rnorm<1.15*sigma):
                

                        #First cluster found
                        if (check==0):


                            clusters[j,0]=clusters[j,0]+1     #Increment size of cluster j
                            clusters[j,clusters[j,0]]=i       #Adding i to the cluster

                        
                            #Deleting the cluster corresponding to particle i 
                            clusters[i,:]=-1
                            clusters[i,0]=0
                        

                            check=1
                            main=j
                            break
                        #Particle i belongs to another cluster as well
                        if (check==1):

                        
                            #Merge this cluster j with main cluster found earlier
                            ini=clusters[main,0]
                        
                            clusters[main,0]=clusters[main,0]+clusters[j,0]     #Updating size of main
                        
                            clusters[main,ini+1:clusters[main,0]+1]=clusters[j,1:clusters[j,0]+1]  #Including elements

                            #Delete cluster j
                            clusters[j,:]=-1
                            clusters[j,0]=0
                            
                
    return(clusters)

