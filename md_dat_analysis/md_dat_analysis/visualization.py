import os
from md_dat_analysis import cluster_analysis as cla


#Prepare xyz file to Visualize 2D particle system snapshots in OVITO
def ovito2d(pos,colour2,path,filename,length):
    
    os.chdir(path)
    file = open(filename,'w')

    #3D array: trajectory file
    if (len(pos.shape)==3):
        for i in range(0,pos.shape[2]-1):
            file.write('{}\n'.format(pos.shape[0]))
            file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
            for j in range(pos.shape[0]):
            
                if (j in colour2):
                    file.write('2 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))
            
                else:
                    file.write('1 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))
                    
    
    #2D array: Only a single frame
    if (len(pos.shape)==2):
        file.write('{}\n'.format(pos.shape[0]))
        file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
        for j in range(pos.shape[0]):
            
            if (j in colour2):
                file.write('2 {} {}\n'.format(pos[j,0],pos[j,1]))
            
            else:
                file.write('1 {} {}\n'.format(pos[j,0],pos[j,1]))

    
    
    file.close()
    
    
    
#Prepare xyz file to Visualize 3D particle system snapshots in OVITO
def ovito3d(pos,colour2,path,filename,length):
    
    os.chdir(path)
    file = open(filename,'w')

    #3D array: trajectory file
    if (len(pos.shape)==3):
        for i in range(0,pos.shape[2]-1):
            file.write('{}\n'.format(pos.shape[0]))
            file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
            for j in range(pos.shape[0]):
            
                if (j in colour2):
                    file.write('2 {} {} {}\n'.format(pos[j,0,i],pos[j,1,i],pos[j,2,i]))
            
                else:
                    file.write('1 {} {} {}\n'.format(pos[j,0,i],pos[j,1,i],pos[j,2,i]))
                    
    
    #2D array: Only a single frame
    if (len(pos.shape)==2):
        file.write('{}\n'.format(pos.shape[0]))
        file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
        for j in range(pos.shape[0]):
            
            if (j in colour2):
                file.write('2 {} {} {}\n'.format(pos[j,0],pos[j,1],pos[j,2]))
            
            else:
                file.write('1 {} {} {}\n'.format(pos[j,0],pos[j,1],pos[j,2]))

    
    
    file.close()




#Prepare xyz of snapshots colour coded accoring to 4.6 abop and whether they are fluid or disordered
def ovito2d_4states(pos,path,filename,nparticles,ndims,sigma,length,hlength):
    os.chdir(path)
    
    file = open(filename,'w')
    

    

    
    #2D array: Only a single frame
    if (len(pos.shape)==2):
        
        z,nc,nclist,r_list =cla.coord_number(nparticles,ndims,sigma,length,hlength,pos)
        phi4_avg,phi4=cla.abop(nparticles,ndims,4,r_list,nc)    
        phi6_avg,phi6=cla.abop(nparticles,ndims,6,r_list,nc)
        file.write('{}\n'.format(pos.shape[0]))
        file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
        
        for i in range(pos.shape[0]):
            
            
            #Fluid Condition
            if (nc[i]<=2):
                file.write('1 {} {}\n'.format(pos[i,0],pos[i,1]))

            #HCP Condition 
            elif ((phi6[i]>=0.7)&(phi4[i]<=0.3)):
                file.write('2 {} {}\n'.format(pos[i,0],pos[i,1]))

            #Quadratic Condition
            elif ((phi4[i]>=0.7)&(phi6[i]<=0.3)):
                file.write('3 {} {}\n'.format(pos[i,0],pos[i,1]))

            #None above then disordered
            else:
                file.write('4 {} {}\n'.format(pos[i,0],pos[i,1]))
                
                
    if (len(pos.shape)==3):
        for i in range(0,pos.shape[2]-1,100):
            
            z,nc,nclist,r_list =cla.coord_number(nparticles,ndims,sigma,length,hlength,pos[:,:,i])
            phi4_avg,phi4=cla.abop(nparticles,ndims,4,r_list,nc)    
            phi6_avg,phi6=cla.abop(nparticles,ndims,6,r_list,nc)
            
            file.write('{}\n'.format(pos.shape[0]))
            file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
        
            for j in range(pos.shape[0]):
            
            
                #Fluid Condition
                if (nc[j]<=2):
                    file.write('1 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))

                #HCP Condition 
                elif ((phi6[j]>=0.7)&(phi4[j]<=0.3)):
                    file.write('2 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))

                #Quadratic Condition
                elif ((phi4[j]>=0.7)&(phi6[j]<=0.3)):
                    file.write('3 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))

                #None above then disordered
                else:
                    file.write('4 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))
            
    
    file.close()
    
    

#Function to colour code particles according to abop phi_4 and phi_6
def ovito2d_abop(pos,path,filename,nparticles,ndims,sigma,length,hlength):
    os.chdir(path)
    file = open(filename,'w')
    
    
    #2D array: Only a single frame
    if (len(pos.shape)==2):
        z,nc,nclist,r_list =cla.coord_number(nparticles,ndims,sigma,length,hlength,pos)
        phi4_avg,phi4=cla.abop(nparticles,ndims,4,r_list,nc)    
        phi6_avg,phi6=cla.abop(nparticles,ndims,6,r_list,nc)
        file.write('{}\n'.format(pos.shape[0]))
        file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
        
        for i in range(pos.shape[0]):
            file.write('1 {} {} {} {}\n'.format(pos[i,0],pos[i,1],phi4[i],phi6[i]))
            
    if (len(pos.shape)==3):
        for i in range(0,pos.shape[2]-1,100):
            z,nc,nclist,r_list =cla.coord_number(nparticles,ndims,sigma,length,hlength,pos)
            phi4_avg,phi4=cla.abop(nparticles,ndims,4,r_list,nc)    
            phi6_avg,phi6=cla.abop(nparticles,ndims,6,r_list,nc)
            file.write('{}\n'.format(pos.shape[0]))
            file.write('Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(length,length,length))
            
            for j in range(pos.shape[0]):
                file.write('1 {} {} {} {}\n'.format(pos[j,0],pos[j,1],phi4[j],phi6[j]))
                
    file.close()