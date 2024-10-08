#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
import time
import fcl
import util

from numpy.random import default_rng

from torch.utils.data import TensorDataset, DataLoader, Dataset
from scipy.spatial.transform import Rotation as R

from torch.optim import Adam, SGD
from pointnet2_utils import PointNetSetAbstraction

from urdfpy import URDF as pURDF
import trimesh
import pytorch_kinematics as pk
import pytorch3d.transforms.rotation_conversions as rc

active_link_names=["right_shoulder_fe_link","right_shoulder_aa_link","right_shoulder_ie_link","right_elbow_fe_link","right_wrist_rotation_link","right_wrist_flexion_link"]
active_joint_names=["right_shoulder_fe","right_shoulder_aa","right_shoulder_ie","right_elbow_fe","right_wrist_rotation","right_wrist_flexion"]

robot=pURDF.load("./lsa1.1_new/robot_obj_combined_hulls.urdf")
sizes=[]
active_link_meshes=[]
for link in active_link_names:
    mesh=trimesh.load("./lsa1.1_new/meshes/urdf/obj_combined_hulls/"+link+".obj")
    active_link_meshes.append(mesh)
    sizes.append(mesh.bounding_box.extents)


scene_descriptions=np.load("scene_descriptions_3.npy",allow_pickle=True)

chain = pk.build_serial_chain_from_urdf(open("./lsa1.1_new/robot_obj_combined_hulls.urdf").read(), active_link_names[5],"right_shoulder_fe_link")

rand=default_rng()


# In[2]:


class PN(nn.Module):
    def __init__(self,num_class,in_channel=3):
        super(PN, self).__init__()
        if in_channel>3:
            self.additional_channels = True
        else:
            self.additional_channels= False

        self.sa1 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=in_channel, mlp=[1024, 1024,2048, 2048,1024,512,64], group_all=True)

        self.lin1 = nn.Linear(64, 256)

        self.gn1=nn.GroupNorm(16,64)
        self.gn2=nn.GroupNorm(16,256)

        #self.clin1=nn.Linear(1024,1)

        self.ss_flin=nn.Linear(256,num_class)

    def forward(self,xyz):
        B, _, _ = xyz.shape
        if self.additional_channels:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz,norm)
        pn_out = F.relu(self.gn2(self.lin1(self.gn1(l1_points.reshape(B, -1)))))

        out=self.ss_flin(pn_out)

        return out, None


# In[3]:


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=PN(12,in_channel=9)
model.load_state_dict(torch.load("new_best_model_arm_jps_sc_2.pth",map_location=device))
model=model.eval()
new_model=torch.compile(model)

model=model.to(device)

network_dist_func = lambda x,y:combined_dist_func(model,x,y)

network_dist_func_2 = lambda x,y:network_dist_func_joint(model,x,y)


###Distance Function that takes as input a trained network along with a pointcloud and returns the distance from the robot arm to the pointcloud along with the gradient direction in C-space 
###Of the distance function

def network_dist_func_joint(network,pc,jps,normalize=False,verbose=False):
            
    if normalize:    
        pc_centre=pc.sum(axis=0)/pc.shape[0]
        pc=pc-pc_centre
    
    jps=copy.deepcopy(jps.reshape(-1).type(torch.float))

    jps.requires_grad=True

    to_append=jps.repeat(pc.shape[0],1)

    augmented_pc=torch.cat([pc,to_append],dim=1)

    test_pc=augmented_pc.reshape(1,-1,3+jps.shape[0])

    test_pc=test_pc.permute(0,2,1).to(device)
    
    min_dist_pred,point_ind=network(test_pc)
    
    if verbose:
        print("min distances",min_dist_pred[0])
        
    min_dist_pred=min_dist_pred[0][:6]/100
    
    overall_min_dist=min_dist_pred.min()
    
    link_num=min_dist_pred.argmin()    

    overall_min_dist.backward()
        
    grad_dir=jps.grad
    

    md=overall_min_dist.detach().cpu().numpy()
           
    gd=grad_dir.detach().cpu().numpy()
    
    return md,gd
    
###Does the same as before but also computes the distance with trimesh. This is just so we can get access to the closest point on the arm to the point cloud and use that to visualize the gradient
###direction, not for speed comparisons

def combined_dist_func(network,pc,jps,normalize=False,verbose=False):
    
            
    if normalize:    
        pc_centre=pc.sum(axis=0)/pc.shape[0]
        pc=pc-pc_centre
    
    jps=copy.deepcopy(jps.reshape(-1).type(torch.float))

    jps.requires_grad=True

    to_append=jps.repeat(pc.shape[0],1)

    augmented_pc=torch.cat([pc,to_append],dim=1)

    test_pc=augmented_pc.reshape(1,-1,3+jps.shape[0])

    test_pc=test_pc.permute(0,2,1).to(device)
    
    min_dist_pred,point_ind=network(test_pc)
    
    if verbose:
        print("mds",min_dist_pred[0])
        
    min_dist_pred=min_dist_pred[0][:6]/100
        
    overall_min_dist=min_dist_pred.min()

    
    link_num=min_dist_pred.argmin().item()    

    overall_min_dist.backward()
        
    grad_dir=jps.grad
    

    md=overall_min_dist.detach().cpu().numpy()
           
    gd=grad_dir.detach().cpu().numpy()
    
    
    poses,meshes,_=util.generate_poses_from_jps(robot,[jps],active_link_names,active_joint_names,sizes)
    
    closest_points_in_pc=[]
    closest_points_on_meshes=[]
    mds=[]
    for mesh in meshes[0]:        

        closest_points,dist,_=trimesh.proximity.closest_point(mesh,pc)
    
        closest_point_ind=np.argmin(dist)
    
        closest_points_on_meshes.append(closest_points[closest_point_ind])
    
        closest_points_in_pc.append(pc[closest_point_ind])
                
        mds.append(dist[closest_point_ind])
        
    closest_link=np.argmin(mds)
    
    position=poses[0][closest_link][:3]
    closest_point_in_pc=closest_points_in_pc[closest_link]
    closest_point_on_mesh_tm=closest_points_on_meshes[closest_link]
    closest_point_on_mesh_net=closest_points_on_meshes[link_num]
    tm_gd=closest_point_in_pc-closest_point_on_mesh_tm
    
    return md,gd,link_num,closest_point_on_mesh_tm,closest_point_on_mesh_net,tm_gd


###Control distance function that always returns infinity, used to see the path without adjustment 

def null_dist_func(pc,pose):
    return np.inf,np.array([])

####Takes a gradient direction in workspace and uses pytorch kinematics to compute the corresponding gradient in C-space

def workspace_grad_to_cspace_grad(grad,link,jps):
        
    tjps=torch.tensor(jps).type(torch.float)
    tjps.requires_grad=True
    ret = chain.forward_kinematics(tjps, end_only=False)
    m=ret[active_link_names[link]].get_matrix()[0]

    pos = m[:3, 3]

    rot_e=rc.matrix_to_euler_angles(m[:3,:3],'XYZ')

    o=torch.cat([pos,rot_e])
        
    o.backward(torch.tensor(grad).type(torch.float))
    jps_grad=tjps.grad
        
    return jps_grad


###Computes the distance and gradient using trimesh, it does this by finding the closest point on the arm to the pointcloud along with the closest point in the pointcloud and using the 
###Vector between them as a gradient in workspace which it then converts to a gradient in C-space

def trimesh_dist_func(pc,jps,rot_dir=False):
    
    
    poses,meshes,_=util.generate_poses_from_jps(robot,[jps],active_link_names,active_joint_names,sizes)
    
    closest_points_in_pc=[]
    closest_points_on_meshes=[]
    mds=[]
    for mesh in meshes[0]:        

        closest_points,dist,_=trimesh.proximity.closest_point(mesh,pc)
    
        closest_point_ind=np.argmin(dist)
    
        closest_points_on_meshes.append(closest_points[closest_point_ind])
    
        closest_points_in_pc.append(pc[closest_point_ind])
            
        mds.append(dist[closest_point_ind])
                
    closest_link=np.argmin(mds)
    
    position=poses[0][closest_link][:3]

    rot=R.from_quat(poses[0][closest_link][3:7]).as_matrix()
    eul_rot=R.from_quat(poses[0][closest_link][3:7]).as_euler('xyz')
    rot_inv_mat=np.linalg.inv(rot)
    
    closest_point_in_pc=closest_points_in_pc[closest_link]
    closest_point_on_mesh=closest_points_on_meshes[closest_link]
    md=mds[closest_link]
    
    gd=-closest_point_in_pc+closest_point_on_mesh
    
    gd=gd.type(torch.float)
    
    
    ###CALCULATE ROT DIR
    
    p=closest_point_in_pc-position
    rp=closest_point_on_mesh-position
    
    p_init=np.matmul(rot_inv_mat,rp)
    
    if rot_dir:
    
        rot_dir=torch.tensor(np.array(util.rot_dir_from_point(torch.tensor(p).type(torch.float),torch.tensor(p_init).type(torch.float),eul_rot))).type(torch.float)
    else:
        rot_dir=torch.tensor([0,0,0]).type(torch.float)
    
    
    ###CALCULATE GRADIENT IN C-SPACE
    
    jps_grad=workspace_grad_to_cspace_grad(torch.cat([gd,rot_dir]),closest_link,jps)
    
    return md,jps_grad.detach().cpu().numpy()

    
def get_jacobian(jps,offset,ee_link):
    
    chain = pk.build_serial_chain_from_urdf(open("./lsa1.1_new/robot_obj_combined_hulls.urdf").read(), active_link_names[ee_link],"right_shoulder_fe_link")
    
    J = chain.jacobian(jps, locations=offset)
    return J    
    


######## Adjust Arm Path###########
#Given a path in a scene and a distance function, try to move the robot arm along the path, if the distance function
#detects that the robot arm is too close to an object in the scene adjust the trajectory using the gradient returned
#by the distance function in order to avoid the object

def path_adjust(scene,jps,dist_func,motion_scale=0.02,dist_threshold=0.02,scale_adjust=1,max_tries=200):
    
    distances=[0]
    path_poses=[]
    grad_directions=[]
    current_dist=0
    num_collisions=0
    
    final_jps=jps[-1]
    current_jps=jps[0]
    path_poses.append(current_jps)
    
    test_pose,_,_=util.generate_poses_from_jps(robot,[final_jps],active_link_names,active_joint_names,sizes)
    
    goal_ee_position=test_pose[0,-1,:3]

    scene_meshes,ta=util.create_meshes(scene)
    pc=util.create_pc(scene_meshes,5000,ta)

    total_checkpoints=jps.shape[0]

    md,gd=get_grad(pc,dist_func,current_jps)
    grad_directions.append(gd)
    
    current_dist,current_ee_pos=check_goal(jps[0],jps[-1])
    
    net_ee_movement=0
    
    previous_ee_pos=current_ee_pos
    
    next_checkpoint=1
    next_jps=jps[next_checkpoint]
    
    current_closest_jps=copy.deepcopy(current_jps)

    closest_next_checkpoint_dist,_=check_goal(current_jps,next_jps)

    num_tries=0
    new_path=True
    num_tries_on_checkpoint=0
    while current_dist>0.02 and num_tries<max_tries and next_checkpoint<jps.shape[0]:
        rerouting=False
        num_tries+=1    
        
        direction_vector=next_jps-current_jps
                
        norm_dv=direction_vector/np.linalg.norm(direction_vector)
        
        print("md detect",md)
        if md<=0:

            num_collisions+=1
            md=0    
            
        if gd.any():  
            
            gd=torch.from_numpy(gd)
            norm_gd=gd/torch.linalg.norm(gd)  
            

        
            if md<dist_threshold and gd.any():
                rerouting=True

                
                if  torch.dot(norm_dv,norm_gd)<-0.8:

                    orth_vec=torch.tensor([1.,1.,1.,1.,1.,1.])
                    orth_vec-=orth_vec.dot(norm_dv)*norm_dv
                    orth_vec/=torch.linalg.norm(orth_vec)

                    if np.random.rand()>0.5:

                        norm_dv=orth_vec
                    else:
                        norm_dv=-1*orth_vec
                        
                else:

                    norm_gd=norm_gd*scale_adjust

                    direction_vector=norm_dv+norm_gd

                    norm_dv=direction_vector/np.linalg.norm(direction_vector)

        norm_dv=norm_dv*motion_scale
            
        current_jps=current_jps+norm_dv
        
        path_poses.append(current_jps)
                   
        next_checkpoint_dist,current_ee_pos=check_goal(current_jps,next_jps)
        
        if next_checkpoint_dist<closest_next_checkpoint_dist:
            closest_jps=copy.deepcopy(current_jps)
    
        if torch.linalg.norm(current_closest_jps-current_jps)>0.001:
            num_tries_on_checkpoint=0

        else:
            num_tries_on_checkpoint+=1
            
        md,gd=get_grad(pc,dist_func,current_jps)

        grad_directions.append(gd)
        
        current_dist,current_ee_pos=check_goal(current_jps,next_jps)    
        
        if next_checkpoint_dist<0.02 or num_tries_on_checkpoint>10:
            print("num tries on checkpoint",num_tries_on_checkpoint)
            next_checkpoint+=1

            if next_checkpoint>=total_checkpoints:
                break
            
            next_jps=jps[next_checkpoint]

            num_tries_on_checkpoint=0
            closest_next_checkpoint_dist=check_goal(current_jps,next_jps)
                
    success= current_dist<0.02
    
    print("num tries",num_tries)
    print("distance",current_dist)

    return num_collisions, success, distances,current_dist,path_poses,grad_directions


######Same function as before but also record the nearest points between the robot and the scene, this expects you
######To use combined_dist_func which computes this information as well, this will be slower but allow you to visualize 
######The gradient direction


def path_adjust_verbose(scene,jps,dist_func,motion_scale=0.02,dist_threshold=0.02,scale_adjust=1,max_tries=200):
    
    distances=[0]
    path_poses=[]
    grad_directions=[]
    closest_links=[]
    closest_points_tm=[]
    closest_points_net=[]
    tmgds=[]
    current_dist=0
    num_collisions=0
    
    final_jps=jps[-1]
    current_jps=jps[0]
    path_poses.append(current_jps)
    
    test_pose,_,_=util.generate_poses_from_jps(robot,[final_jps],active_link_names,active_joint_names,sizes)
    
    goal_ee_position=test_pose[0,-1,:3]

    scene_meshes,ta=util.create_meshes(scene)
    pc=util.create_pc(scene_meshes,5000,ta)

    total_checkpoints=jps.shape[0]

    md,gd,closest_link,closest_point_tm,closest_point_net,tmgd=get_grad_verbose(pc,dist_func,current_jps)
    closest_links.append(closest_link)
    closest_points_tm.append(closest_point_tm)
    closest_points_net.append(closest_point_net)
    tmgds.append(tmgd)
    grad_directions.append(gd)
    
    current_dist,current_ee_pos=check_goal(jps[0],jps[-1])
    
    net_ee_movement=0
    
    previous_ee_pos=current_ee_pos
    
    next_checkpoint=1
    next_jps=jps[next_checkpoint]
    
    current_closest_jps=copy.deepcopy(current_jps)

    closest_next_checkpoint_dist,_=check_goal(current_jps,next_jps)

    num_tries=0
    new_path=True
    num_tries_on_checkpoint=0
    while current_dist>0.02 and num_tries<max_tries and next_checkpoint<jps.shape[0]:
        rerouting=False
        num_tries+=1    
        
        direction_vector=next_jps-current_jps
                
        norm_dv=direction_vector/np.linalg.norm(direction_vector)
        
        print("md detect",md)
        if md<=0:
            num_collisions+=1
            md=0    
            
        if gd.any():  
            
            gd=torch.from_numpy(gd)
            norm_gd=gd/torch.linalg.norm(gd)  
            

        
            if md<dist_threshold and gd.any():
                rerouting=True
                
                if  torch.dot(norm_dv,norm_gd)<-0.8:
                    
                    orth_vec=torch.tensor([1.,1.,1.,1.,1.,1.])
                    orth_vec-=orth_vec.dot(norm_dv)*norm_dv
                    orth_vec/=torch.linalg.norm(orth_vec)

                    if np.random.rand()>0.5:

                        norm_dv=orth_vec
                    else:
                        norm_dv=-1*orth_vec
                        
                else:

                    norm_gd=norm_gd*scale_adjust


                    direction_vector=norm_dv+norm_gd

                    norm_dv=direction_vector/np.linalg.norm(direction_vector)

        norm_dv=norm_dv*motion_scale
       
        
        current_jps=current_jps+norm_dv
        
        path_poses.append(current_jps)
                   
        next_checkpoint_dist,current_ee_pos=check_goal(current_jps,next_jps)
        
        if next_checkpoint_dist<closest_next_checkpoint_dist:
            closest_jps=copy.deepcopy(current_jps)
    
        if torch.linalg.norm(current_closest_jps-current_jps)>0.001:
            num_tries_on_checkpoint=0

        else:
            num_tries_on_checkpoint+=1
            
        md,gd,closest_link,closest_point_tm,closest_point_net,tmgd=get_grad_verbose(pc,dist_func,current_jps)
        closest_links.append(closest_link)
        tmgds.append(tmgd)
        closest_points_tm.append(closest_point_tm)
        closest_points_net.append(closest_point_net)
        grad_directions.append(gd)
        
        current_dist,current_ee_pos=check_goal(current_jps,next_jps)    
        
        
        if next_checkpoint_dist<0.02 or num_tries_on_checkpoint>10:
            print("num tries on checkpoint",num_tries_on_checkpoint)
            next_checkpoint+=1

            if next_checkpoint>=total_checkpoints:
                break
            
            next_jps=jps[next_checkpoint]

            num_tries_on_checkpoint=0
            closest_next_checkpoint_dist=check_goal(current_jps,next_jps)
                
    success= current_dist<0.02
    
    print("num tries",num_tries)
    print("distance",current_dist)

    return num_collisions, success, distances,current_dist,path_poses,grad_directions,closest_links,closest_points_tm,closest_points_net,tmgds,pc


def get_grad(pc,dist_func,jps):
    
    nogpc=util.gen_local_pc(robot,active_link_names,active_joint_names,sizes,jps,pc)
            
    md,gd=dist_func(nogpc,jps)
            
    return md,gd


def get_grad_verbose(pc,dist_func,jps):
    
    nogpc=util.gen_local_pc(robot,active_link_names,active_joint_names,sizes,jps,pc)
            
    md,gd,closest_link,closest_point_tm,closest_point_net,tmgd=dist_func(nogpc,jps)
            
    return md,gd,closest_link,closest_point_tm,closest_point_net,tmgd

def check_goal(jps_i,jps_f):
    
    test_pose,_,_=util.generate_poses_from_jps(robot,[jps_i,jps_f],active_link_names,active_joint_names,sizes)
    
    current_ee_position=test_pose[0,-1,:3]
    final_ee_position=test_pose[1,-1,:3]
    
    
    return np.linalg.norm(final_ee_position-current_ee_position),current_ee_position



#### Example Scene

new_scene=[[0.7758353634624906,
 0.0,
 0.12956332002280435,
 1.0,
 0.0,
 0.0,
 0.0,
 1.1715150053480243,
 1.2594135016013932,
 0.5],
[0.4230324521196277,
 -0.2722247421121958,
 0.45833856513879195,
 -0.024467546477714628,
 -0.024467546477714628,
 -0.24379382944663586,
 -0.5040490190576641,
 0.16677285224514682,
 0.18288695242856023,
 0.1575504902319752]]
   
new_jps=[1.8,
-1.5,
-0.5566179021071089,
-0.04272648694721992,
0.6865317756165505,
0.224307393770238]

   
new_jps_2=[1.8,
-0.1,
-0.5566179021071089,
-0.04272648694721992,
0.6865317756165505,
0.224307393770238]


# In[64]:


import time
motion_scale=0.005
dist_threshold=0.02
scale_factor=10000

average_time=0
times=[]
goals_reached=[]
collisions=[]
num_success=0
num_collide=0
num_examples=1


for i in range(3,4):
    print("example", i)
    
    cjps_1=torch.tensor(new_jps).type(torch.float).reshape(1,-1)
    cjps_2=torch.tensor(new_jps_2).type(torch.float).reshape(1,-1)
    joint_ps=torch.cat([cjps_1,cjps_2])
    scene=new_scene   
    
    #cjps_1=torch.tensor(jpss[i][0]).type(torch.float).reshape(1,-1)
    #cjps_2=torch.tensor(jpss[i][-1]).type(torch.float).reshape(1,-1)  
        
    #joint_ps=torch.cat([cjps_1,cjps_2])
    #scene=new_scenes[i][:]

    t0=time.time()
    n,s,d,cd,jps_path_4,grad_dirs_4,closest_links_4,closest_points_tm_4,closest_points_net_4,tmgds_4,pc=path_adjust_verbose(scene,joint_ps,network_dist_func,motion_scale=motion_scale,dist_threshold=dist_threshold,scale_adjust=scale_factor,max_tries=30)
    #n,s,d,cd,jps_path_3,grad_dirs=path_adjust(scene,joint_ps,network_dist_func_2,motion_scale=motion_scale,dist_threshold=dist_threshold,scale_adjust=scale_factor,max_tries=100)
    example_time=time.time()-t0
    times.append(example_time)
    
    goals_reached.append(s)
    
    average_time+=example_time
    
    scene_meshes,_=util.create_meshes(scene)
    scene_objs=[]

    for j in range(len(scene_meshes)):

        mesh=scene_meshes[j]

        geom=trimesh.collision.mesh_to_BVH(mesh)
        obj=fcl.CollisionObject(geom)
        scene_objs.append(obj)

    num_true_collisions=0    
    
    test_poses,test_meshes,test_transforms=util.generate_poses_from_jps(robot,jps_path_4,active_link_names,active_joint_names,sizes)
    
    for link_meshes in test_meshes:
        true_min_dist_pose=np.inf
        for link_mesh in link_meshes:
            

            geom=trimesh.collision.mesh_to_BVH(link_mesh)
            obj=fcl.CollisionObject(geom)
            md,_,point2,_,_=util.calculate_distance(scene_objs,obj,penetration_depth=True)
            true_min_dist_pose=np.min([true_min_dist_pose,md])
        print("md actual",true_min_dist_pose)
        if true_min_dist_pose<0:
            num_true_collisions+=1
            
    collision=np.min([num_true_collisions,1])
    collisions.append(collision)
    
    if collision==0 and s:
        num_success+=1
        
average_time=average_time/num_examples



pos_grad_dirs=[]

for l in range(len(grad_dirs_4)):
    closest_link=closest_links_4[l]
    closest_point=closest_points_net_4[l]
    gd=grad_dirs_4[l][:closest_link+1]
    J=get_jacobian(gd,closest_point,closest_link)

    pos_grad_dirs.append(torch.matmul(J,torch.tensor(gd))[0][:3])
    
for l in range(len(pos_grad_dirs)):
    pos_grad_dirs[l]=pos_grad_dirs[l]*-1



grm_net_3=util.create_grad_rot_matricies(pos_grad_dirs,closest_points_net_4)
grm_tm_3=util.create_grad_rot_matricies(tmgds_4,closest_points_tm_4)


test_traj_8=util.generate_poses_from_jps(robot,jps_path_4,active_link_names,active_joint_names,sizes)
util.animate_traj(scene[:],test_traj_8[2],test_traj_8[1][0],grm_tm_3,sizes,fps=10,obj_type='box',show_frame=False,pc=None)

