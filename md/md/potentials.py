import numpy as np
from numba import jit,prange
from md import pbc
from numba.np.extensions import cross2d







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
    






#Forces due to Lennard-Jones potential (optimized for single core)
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





#Forces due to Lennard-Jones potential (optimized for multiple core)
@jit(nopython=True,parallel=True)
def lj_force_p(pos,lj_params,shape_params,pbc_params,nc,nclist):

    force=np.zeros((shape_params[0],shape_params[1]),dtype=np.float64)     #Force matrix nparticles x ndims
    r=np.zeros((shape_params[1]),dtype=np.float64)                         #r to hold the distance between particle paris as a vector


    
    #Loop running over all neighbours (neighbours here contain all neighbours)
    for i in prange(shape_params[0]):
        for j in range(1,nc[i]+1):
            p=nclist[i,j]
            
            
            #Minimum image criteria distance computation
            rnorm,r = pbc.dist_mic(shape_params[1],pos[i,:],pos[p,:],pbc_params[0],pbc_params[1])
            
            if rnorm<lj_params[2]:
                part=(lj_params[1]/rnorm)**6
                f=(-24.0*lj_params[0]/rnorm)*(2*part**2-part)
         
                for k in range(shape_params[1]):
                    force[i,k]=force[i,k]+(r[k]/rnorm)*f

    return(force)






#Janus particle translational attractive potential & forces(gradient with respect to r_ij) (as described in Mallory et. at) 
@jit(nopython=True)
def janus_attractive(lj_params, patch_params, rnorm):
    r_s = np.abs(rnorm-lj_params[1])                            # distance between particles surfaces
    bracket = patch_params[3]/(r_s+patch_params[3]*2**(1/6))
    
    # attractive potential
    U_att = 4*patch_params[0]*(bracket**12-bracket**6)
    # attractive force
    F_att = 24*patch_params[0]*(bracket**6 - 2*bracket**12)*((rnorm-lj_params[1])/(r_s*(r_s+patch_params[3]*2**(1/6))))
    return U_att, F_att



#Janus particle translational repulsive potential & forces(gradient with respect to r_ij) (as described in Mallory et. at) 
@jit(nopython=True)
def janus_repulsive(lj_params, rnorm):
    # repulsive potential
    U_rep = 4*lj_params[0]*(lj_params[1]/rnorm)**12
    # repulsive force
    F_rep = 12*U_rep*(-1/rnorm)
    return F_rep, U_rep




#Angular interaction potential and gradients with respect to theta and r_ij (as described in Mallory et. at) 
@jit(nopython=True)
def janus_angular(direction, patch_params, r, rnorm, particle_type):

    # definitions
    phi = 1.0
    torque = 0.0
    upper_limit = patch_params[1]+patch_params[2]      # theta_max + theta_tail
    lower_limit = patch_params[1]                      # theta_max

    # normalized inter particle vector
    r_ip = r/rnorm
    
    # determining the angle
    cos_theta = np.dot(r_ip, direction)
    abs_theta = np.arccos(cos_theta)

    # checking for farticle type; if 'double' and angle bigger than 90° the opposite patch is nearer
    if particle_type == 'double':
        if abs_theta > (np.pi)/2:
            abs_theta = np.abs(abs_theta-np.pi)
            direction = (-1)*direction

    # compute potential and torque if angle is within the interaction range
    if abs_theta >= lower_limit and abs_theta <= upper_limit:

        # determining the sign via the 2d cross rpoduct
        sign = cross2d(direction,r_ip).item()/np.sin(abs_theta)
        sign = np.round(sign)             

        # argument inside the cos-function of the angular potential
        arg = np.pi*(abs_theta-patch_params[1])/(2*patch_params[2])  

        phi = (np.cos(arg))**2
        torque = np.pi*(np.cos(arg)*np.sin(arg))/patch_params[2]
        
        # The components of the translational force in patch direction (_n) and in inter-particle-direction (_r)
        nabla_phi_n = torque*1/(np.sqrt(1-(np.dot(r_ip,direction))**2)*rnorm)
        nabla_phi_r = nabla_phi_n*(np.dot(r,direction)/rnorm)

    if abs_theta > upper_limit:
        phi = 0.0

    return phi, torque, sign, direction, nabla_phi_n, nabla_phi_r






#Compute forces on particles due to a single hydrophobic patch Janus particle ([1] S. A. Mallory, 
# F. Alarcon, A. Cacciuto, and C. Valeriani, Self-Assembly of Active Amphiphilic Janus Particles, New J. Phys. 19, 125014 (2017).
@jit(nopython=True)
def single_patch_force(pos, direction, lj_params, shape_params, pbc_params, patch_params, nc, nclist):

    force=np.zeros((shape_params[0],shape_params[1]),dtype=np.float64)     
    torque=np.zeros((shape_params[0]),dtype=np.float64)
    r=np.zeros((shape_params[1]),dtype=np.float64)                         


    for i in range(shape_params[0]):
        for j in range(1,nc[i]+1):
            p=nclist[i,j]
            
            
            rnorm,r = pbc.dist_mic(shape_params[1],pos[i,:],pos[p,:],pbc_params[0],pbc_params[1])
        

            if rnorm<lj_params[2]:

                phi_i, torque_i, sign_i, direction_i, nabla_phi_i_n, nabla_phi_i_r = janus_angular(direction[i], patch_params, (-1)*r, rnorm, particle_type='single')
                phi_p, torque_p, sign_p, direction_p, nabla_phi_p_n, nabla_phi_p_r = janus_angular(direction[p], patch_params, r, rnorm, particle_type='single')

                F_rep, _ = janus_repulsive(lj_params, rnorm)

                U_att, F_att = janus_attractive(lj_params, patch_params, rnorm)

                F_r = F_rep + F_att*phi_i*phi_p + U_att*(phi_p*nabla_phi_i_r + phi_i*nabla_phi_p_r)
                F_n = U_att*(phi_p*nabla_phi_i_n*direction_i + phi_i*nabla_phi_p_n*direction_p)

                torque[i] += U_att*sign_i*torque_i*phi_p
                torque[p] += U_att*sign_p*torque_p*phi_i
                
                for k in range(shape_params[1]):
                    force[i,k] += F_n[k] + (r[k]/rnorm)*F_r 
                    force[p,k] -= F_n[k] + (r[k]/rnorm)*F_r 

    return(force, torque)






#Compute forces on particles due to a double hydrophobic patch Janus particle. ([1] S. A. Mallory and A. Cacciuto, 
# Activity-Enhanced Self-Assembly of a Colloidal Kagome Lattice, J. Am. Chem. Soc. 141, 2500 (2019)
@jit(nopython=True)
def double_patch_force(pos, direction, lj_params, shape_params, pbc_params, patch_params, nc, nclist):

    force=np.zeros((shape_params[0],shape_params[1]),dtype=np.float64)     
    torque=np.zeros((shape_params[0]),dtype=np.float64)
    r=np.zeros((shape_params[1]),dtype=np.float64)                         


    #Loop running over all neighbour pairs only
    for i in range(shape_params[0]):
        for j in range(1,nc[i]+1):
            p=nclist[i,j]
            
            
            #Minimum image criteria distance computation
            rnorm,r = pbc.dist_mic(shape_params[1],pos[i,:],pos[p,:],pbc_params[0],pbc_params[1])
        

            if rnorm<lj_params[2]:

                phi_i, torque_i, sign_i, direction_i, nabla_phi_i_n, nabla_phi_i_r = janus_angular(direction[i], patch_params, (-1)*r, rnorm, particle_type='double')
                phi_p, torque_p, sign_p, direction_p, nabla_phi_p_n, nabla_phi_p_r = janus_angular(direction[p], patch_params, r, rnorm, particle_type='double')

                F_rep, _ = janus_repulsive(lj_params, rnorm)

                U_att, F_att = janus_attractive(lj_params, patch_params, rnorm)

                F_r = F_rep + F_att*phi_i*phi_p + U_att*(phi_p*nabla_phi_i_r + phi_i*nabla_phi_p_r)
                F_n = U_att*(phi_p*nabla_phi_i_n*direction_i + phi_i*nabla_phi_p_n*direction_p)

                torque[i] += U_att*sign_i*torque_i*phi_p
                torque[p] += U_att*sign_p*torque_p*phi_i
                
                for k in range(shape_params[1]):
                    force[i,k] += (F_n[k] + (r[k]/rnorm)*F_r)
                    force[p,k] -= (F_n[k] + (r[k]/rnorm)*F_r)

    return(force, torque)