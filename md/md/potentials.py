import numpy as np
from numba import jit,objmode
from md import pbc

# Yukawa Potential (as described in Koegler paper)
@jit(nopython=True)
def potential_yu(nparticles,ndims,pos,e_pos,m_pos,pbc_params,kappa,lj_params):
    r_e=np.zeros((ndims),dtype=np.float64)
    r_m=np.zeros((ndims),dtype=np.float64)
    
    potentialfin=0

    count=0
    
    for i in range(nparticles):
        for j in range(i+1,nparticles):
            for l in range(2):
                for m in range(2):
                    
                    if (l==m):
                        q=1.0
                    if (l!=m):
                        q=-1.0
                    
                    #Minimum image distance computation
                    rnorm_e,r_e = pbc.dist_mic(ndims,e_pos[i,l,:],e_pos[j,m,:],pbc_params[0],pbc_params[1])
                    rnorm_m,r_m = pbc.dist_mic(ndims,m_pos[i,l,:],m_pos[j,m,:],pbc_params[0],pbc_params[1])
                    
                    potential_e=(q*(2.5)**2*(lj_params[0]/lj_params[1])*np.exp(-kappa*rnorm_e))/rnorm_e
                    potential_m=(q*(2.5)**2*(lj_params[0]/lj_params[1])*np.exp(-kappa*rnorm_m))/rnorm_m
                    

                    
                    potentialfin=potentialfin+potential_e+potential_m
                    
                    
    return(potentialfin)





#Forces due to Yukawa potential (as described in Koegler paper)
@jit(nopython=True)
def yu_force(pos,shape_params,e_pos,m_pos,pbc_params,nc,nclist,yu_params,lj_params):
    r_e=np.zeros((shape_params[1]),dtype=np.float64)
    r_m=np.zeros((shape_params[1]),dtype=np.float64)
    

    
    total_f=np.zeros((shape_params[0],shape_params[1]),dtype=np.float64)

    
    for i in range(shape_params[0]):
        for j in range(1,nc[i]+1):
            p=nclist[i,j]
            for l in range(2):
                for m in range(2):
                    
                    if (l==m):
                        q=1.0
                    if (l!=m):
                        q=-1.0
                    
                    #Minimum image distance computation
                    rnorm_e,r_e = pbc.dist_mic(shape_params[1],e_pos[i,l,:],e_pos[p,m,:],pbc_params[0],pbc_params[1])
                    rnorm_m,r_m = pbc.dist_mic(shape_params[1],m_pos[i,l,:],m_pos[p,m,:],pbc_params[0],pbc_params[1])

                    if (rnorm_e<yu_params[1]):
                        f_e=q*(2.5)**2*(lj_params[0]/lj_params[1])*\
                        -1*np.exp(-yu_params[0]*rnorm_e)*((yu_params[0]*rnorm_e+1)/rnorm_e**2)*(yu_params[2]/yu_params[3])
                        
                        for k in range(shape_params[1]):
                            total_f[i,k]=total_f[i,k]+(r_e[k]/rnorm_e)*f_e
                            total_f[p,k]=total_f[p,k]-(r_e[k]/rnorm_e)*f_e
                    
                    if (rnorm_m<yu_params[1]):
                        f_m=q*(2.5)**2*(lj_params[0]/lj_params[1])*\
                        -1*np.exp(-yu_params[0]*rnorm_m)*((yu_params[0]*rnorm_m+1)/rnorm_m**2)*(yu_params[2]/yu_params[3])
                        
                        for k in range(shape_params[1]):
                            total_f[i,k]=total_f[i,k]+(r_m[k]/rnorm_m)*f_m
                            total_f[p,k]=total_f[p,k]-(r_m[k]/rnorm_m)*f_m
                    

                    
                    
                    
    return(total_f)


#Function to compute position of charges inside particles YUkawa only (as described in Koegler paper)
@jit(nopython=True)
def qpos(nparticles,ndims,pos,delr):
    e_pos=np.zeros((nparticles,2,ndims),dtype=np.float64)
    m_pos=np.zeros((nparticles,2,ndims),dtype=np.float64)
    
    
    direction_e=np.zeros((nparticles,ndims),dtype=np.float64)
    direction_m=np.zeros((nparticles,ndims),dtype=np.float64)
    
    #2d unit vector Along y axis
    direction_e[:,0]=0     #np.cos(np.pi/2)
    direction_e[:,1]=1     #np.sin(np.pi/2)
    
    #2d unit vector Along x-axis
    direction_m[:,0]=1     #np.cos(0)
    direction_m[:,1]=0     #np.sin(0)
    
    
    
    # e charge
    #+ve charge
    e_pos[:,0,:]=pos[:,:]+delr*direction_e[:,:]
    #-ve charge
    e_pos[:,1,:]=pos[:,:]-delr*direction_e[:,:]
    
    
    # m charge
    #+ve charge
    m_pos[:,0,:]=pos[:,:]+delr*direction_m[:,:]
    #-ve charge
    m_pos[:,1,:]=pos[:,:]-delr*direction_m[:,:]
    
    # for i in range(nparticles):
    #     for j in range(2):
    #         for k in range(ndims):
    #             if (k==0):      #x axis                     
    #                 m_pos[i,j,k]=pos[i,k]
                    
    #                 if (j==0):  # +ve charge 
    #                     e_pos[i,j,k]=pos[i,k]+delr
    #                 if (j==1):  # -ve charge 
    #                     e_pos[i,j,k]=pos[i,k]-delr
                        
    #             if (k==1):       #y axis
    #                 e_pos[i,j,k]=pos[i,k]
                    
    #                 if (j==0):   #+ve charge
    #                     m_pos[i,j,k]=pos[i,k]+delr
    #                 if (j==1):   #-ve charge
    #                     m_pos[i,j,k]=pos[i,k]-delr

    return(e_pos,m_pos)
    




#Function to compute position of charges inside active particles YUkawa only (as described in Koegler paper)
#Aligning e-field charges along direction of propulsion and m- field ones as required
@jit(nopython=True)
def qpos_active(nparticles,ndims,pos,delr,theta):
    e_pos=np.zeros((nparticles,2,ndims),dtype=np.float64)
    m_pos=np.zeros((nparticles,2,ndims),dtype=np.float64)
    
    direction_e=np.zeros((nparticles,ndims),dtype=np.float64)
    direction_m=np.zeros((nparticles,ndims),dtype=np.float64)
    
    direction_e[:,0]=np.cos(theta)
    direction_e[:,1]=np.sin(theta)
    
    direction_m[:,0]=np.cos(theta+np.pi/2)
    direction_m[:,1]=np.sin(theta+np.pi/2)
    
    
    
    # e charge
    #+ve charge
    e_pos[:,0,:]=pos[:,:]+delr*direction_e[:,:]
    #-ve charge
    e_pos[:,1,:]=pos[:,:]-delr*direction_e[:,:]
    
    
    # m charge
    #+ve charge
    m_pos[:,0,:]=pos[:,:]+delr*direction_m[:,:]
    #-ve charge
    m_pos[:,1,:]=pos[:,:]-delr*direction_m[:,:]
    
    return(e_pos,m_pos)
    






#Forces due to Lennard-Jones potential
@jit(nopython=True)
def lj_force(pos,lj_params,shape_params,pbc_params,nc,nclist):

    force=np.zeros((shape_params[0],shape_params[1]),dtype=np.float64)     #Force matrix nparticles x ndims
    r=np.zeros((shape_params[1]),dtype=np.float64)                         #r to hold the distance between particle paris as a vector


    
    #Loop running over all neighbour pairs only
    for i in range(shape_params[0]):
        for j in range(1,nc[i]+1):
            p=nclist[i,j]
            
            
            #Minimum image criteria distance computation
            rnorm,r = pbc.dist_mic(shape_params[1],pos[i,:],pos[p,:],pbc_params[0],pbc_params[1])
            
            if rnorm<lj_params[2]:
                part=(lj_params[1]/rnorm)**6
                f=(-24.0*lj_params[0]/rnorm)*(2*part**2-part)
         
                for k in range(shape_params[1]):
                    force[i,k]=force[i,k]+(r[k]/rnorm)*f
                    force[p,k]=force[p,k]-(r[k]/rnorm)*f

    return(force)