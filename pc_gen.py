import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R


scene_descriptions=np.load("scene_descriptions.npy",allow_pickle=True)

point_clouds=[]
num_points=5000
for k in range(len(scene_descriptions)):
    pc=[]
    cube_meshes=[]
    total_area=0
    for i in range(len(scene_descriptions[k])):
        #points_per_obj=np.floor_divide(num_points,len(scene_descriptions[k]))
        rot=R.from_quat(scene_descriptions[k][i][3:7]).as_matrix()
        size=scene_descriptions[k][i][7:]
        position=scene_descriptions[k][i][:3]
        
        mat=np.zeros([4,4])
        mat[:3,:3]=rot
        mat[:3,3]=position
        mat[3,3]=1
        cube_mesh=trimesh.creation.box(size,mat)
        total_area+=cube_mesh.area
        cube_meshes.append(cube_mesh)
    total_points=0
    for cube_mesh in cube_meshes:
        num_points_on_cube=np.floor(num_points*cube_mesh.area/total_area)
        total_points+=num_points_on_cube
        for point in trimesh.sample.sample_surface(cube_mesh,int(num_points_on_cube))[0].tolist():
            pc.append(point)
    for m in range(int(num_points)-int(total_points)):
        pc.append(trimesh.sample.sample_surface(cube_meshes[np.random.randint(0,len(cube_meshes))],1)[0].tolist()[0])
    
    np.savetxt("~/point_clouds_arm/pc" +str(k) +".txt",np.array(pc))
