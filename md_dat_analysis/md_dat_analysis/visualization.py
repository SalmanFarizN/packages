import os


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