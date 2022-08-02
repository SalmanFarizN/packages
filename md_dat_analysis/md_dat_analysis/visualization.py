import os
from md_dat_analysis import cluster_analysis as cla


#Prepare xyz file to Visualize 2D particle system snapshots in OVITO
def ovito2d(pos,colour2,path,filename):
    
    os.chdir(path)
    file = open(filename,'w')

    #3D array: trajectory file
    if (len(pos.shape)==3):
        for i in range(0,pos.shape[2]-1):
            file.write('{}\n\n'.format(pos.shape[0]))
            for j in range(pos.shape[0]):
            
                if (j in colour2):
                    file.write('2 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))
            
                else:
                    file.write('1 {} {}\n'.format(pos[j,0,i],pos[j,1,i]))
                    
    
    #2D array: Only a single frame
    if (len(pos.shape)==2):
        file.write('{}\n\n'.format(pos.shape[0]))
        for j in range(pos.shape[0]):
            
            if (j in colour2):
                file.write('2 {} {}\n'.format(pos[j,0],pos[j,1]))
            
            else:
                file.write('1 {} {}\n'.format(pos[j,0],pos[j,1]))

    
    
    file.close()
    




#Prepare xyz of snapshots colour coded accoring to 4.6 abop and whether they are fluid or disordered
def ovito2d_abop(pos,path,filename,nparticles,ndims,sigma,length,hlength):
    os.chdir(path)
    
    file = open(filename,'w')
    
    z,nc,nclist,r_list =cla.coord_number(nparticles,ndims,sigma,length,hlength,pos)
    phi4_avg,phi4=cla.abop(nparticles,ndims,4,r_list,nc)    
    phi6_avg,phi6=cla.abop(nparticles,ndims,6,r_list,nc)
    
    file.write('{}\n\n'.format(pos.shape[0]))
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
            
    
    file.close()