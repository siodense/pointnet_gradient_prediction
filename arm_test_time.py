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


from networks import PN_arm as PN
from torch.utils.data import TensorDataset, DataLoader, Dataset
from scipy.spatial.transform import Rotation as R

from torch.optim import Adam, SGD
from pointnet2_utils import PointNetSetAbstraction

from urdfpy import URDF as pURDF
import trimesh
import pytorch_kinematics as pk
import pytorch3d.transforms.rotation_conversions as rc




def network_func(network,pc,jps,normalize=False):
    
    net_pc=pc
                    
    if normalize:    
        pc_centre=net_pc.sum(axis=0)/net_pc.shape[0]
        net_pc=net_pc-pc_centre
    
    jps=copy.deepcopy(jps.reshape(-1).type(torch.float))

    jps.requires_grad=True

    to_append=jps.repeat(net_pc.shape[0],1)

    augmented_pc=torch.cat([net_pc,to_append],dim=1)

    test_pc=augmented_pc.reshape(1,-1,3+jps.shape[0])

    test_pc=test_pc.permute(0,2,1).to(device)

    min_dist_pred,point_ind=network(test_pc)

    min_dist_pred=min_dist_pred[0][:6]/100

    overall_min_dist=min_dist_pred.min()
           
    overall_min_dist.backward()

    grad_dir=jps.grad

    md=overall_min_dist.detach().cpu().numpy()

    gd=grad_dir.detach().cpu().numpy()
                
    return md,gd

def trimesh_func(pc,jps):
    
    poses,meshes,_=util.generate_poses_from_jps(robot,[jps],active_link_names,active_joint_names,sizes)
    
    closest_points_in_pc=[]
    closest_points_on_meshes=[]
    trimesh_mds=[]
    for mesh in meshes[0]:        

        closest_points,dist,_=trimesh.proximity.closest_point(mesh,pc)
    
        closest_point_ind=np.argmin(dist)
    
        closest_points_on_meshes.append(closest_points[closest_point_ind])
    
        closest_points_in_pc.append(pc[closest_point_ind])
                
        trimesh_mds.append(dist[closest_point_ind])
        
    closest_link=np.argmin(trimesh_mds)

    md=trimesh_mds[closest_link]
    closest_point=closest_points_on_meshes[closest_link]
    closest_point_in_pc=closest_points_in_pc[closest_link]
    workspace_gd=torch.tensor(closest_point)-closest_point_in_pc
    workspace_gd=workspace_gd.type(torch.float)
            
    J=get_jacobian(jps[:closest_link+1],closest_point,closest_link)
    J=J[0][:3][:]
    J_inv=np.linalg.pinv(J)
           
    gd=torch.zeros([6])
    gd[:closest_link+1]=torch.tensor(np.matmul(J_inv,workspace_gd))
    gd=gd.detach().cpu().numpy()

    return md,gd
    
def null_func(pc,jps):
    return np.inf, np.array([])

def get_jacobian(jps,offset,ee_link):
    
    chain = pk.build_serial_chain_from_urdf(open("./lsa1.1_new/robot_obj_combined_hulls.urdf").read(), active_link_names[ee_link],"right_shoulder_fe_link")
    
    J = chain.jacobian(jps, locations=offset)
    return J    
    


######## Adjust Arm Path###########
#Given a path in a scene and a distance function, try to move the robot arm along the path, if the distance function
#detects that the robot arm is too close to an object in the scene adjust the trajectory using the gradient returned
#by the distance function in order to avoid the object. Only records information necessary to complete this task,
#does not include the information needed to visualize the gradient vectors


def path_adjust(scene,jps,dist_func,motion_scale=0.02,dist_threshold=0.02,scale_adjust=1,max_tries=200):
    
    distances=[]
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
    distances.append(md)
    
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

        num_tries+=1    
        
        direction_vector=next_jps-current_jps
                
        norm_dv=direction_vector/np.linalg.norm(direction_vector)
        
        if md<=0:
            num_collisions+=1
            md=0    
            
        if gd.any():  
            
            gd=torch.from_numpy(gd)
            norm_gd=gd/torch.linalg.norm(gd)  
            
            if  torch.dot(norm_dv,norm_gd)<-0.8:
                                    
                orth_vec=torch.tensor([1.,1.,1.,1.,1.,1.])
                orth_vec-=orth_vec.dot(norm_dv)*norm_dv
                orth_vec/=torch.linalg.norm(orth_vec)

                norm_dv+=orth_vec
                

            scale=1/pow((1+10*(md-dist_threshold)),2)
            norm_gd=norm_gd*scale
                    
            if md<dist_threshold:
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
        distances.append(md)
        
        current_dist,current_ee_pos=check_goal(current_jps,next_jps)    
        
        
        if next_checkpoint_dist<0.02 or num_tries_on_checkpoint>10:
            next_checkpoint+=1

            if next_checkpoint>=total_checkpoints:
                break
            
            next_jps=jps[next_checkpoint]

            num_tries_on_checkpoint=0
            closest_next_checkpoint_dist=check_goal(current_jps,next_jps)
                
    success= current_dist<0.02

    return num_collisions, success, distances,current_dist,path_poses,grad_directions,pc


def get_grad(pc,dist_func,jps):
    
    nogpc=util.gen_local_pc(robot,active_link_names,active_joint_names,sizes,jps,pc)
            
    md,gd=dist_func(nogpc,jps)
            
    return md,gd

def check_goal(jps_i,jps_f):
    
    test_pose,_,_=util.generate_poses_from_jps(robot,[jps_i,jps_f],active_link_names,active_joint_names,sizes)
    
    current_ee_position=test_pose[0,-1,:3]
    final_ee_position=test_pose[1,-1,:3]
    
    
    return np.linalg.norm(final_ee_position-current_ee_position),current_ee_position


if __name__=="__main__":

    robot, active_link_names, active_joint_names, sizes=util.get_robot_info()
    
    scene_descriptions=np.load("scene_descriptions_3.npy",allow_pickle=True)
    unique_scenes=[scene_descriptions[i*10] for i in range(1000)]
    motion_trajectories=np.load("motion_trajectories.npy", allow_pickle=True)
    obstructed_examples=np.load("obstructed_examples.npy",allow_pickle=True)

    chain = pk.build_serial_chain_from_urdf(open("./lsa1.1_new/robot_obj_combined_hulls.urdf").read(), active_link_names[5],"right_shoulder_fe_link")

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=PN(12,in_channel=9)
    model.load_state_dict(torch.load("new_best_model_arm_jps_sc_2.pth",map_location=device))
    model=model.eval()
    new_model=torch.compile(model)

    model=model.to(device)

    motion_scale=0.01
    dist_threshold=0.02
    scale_factor=10000

    times={}
    times["tm"]=[]
    times["net"]=[]
    times["none"]=[]

    successes={}
    successes["tm"]=0
    successes["net"]=0
    successes["none"]=0


    dist_funcs={}
    dist_funcs["tm"] = lambda x,y:trimesh_func(x,y)
    dist_funcs["net"]= lambda x,y:network_func(model,x,y)
    dist_funcs["none"] = lambda x,y:null_func(x,y)

    num_examples=obstructed_examples.shape[0]

    for i in range(num_examples):
    
        for method in ["tm","net","none"]:
        
            dist_func=dist_funcs[method]
        
            print("Testing example ", i, " with ", method)

            cjps_1=torch.tensor(motion_trajectories[i][0]).type(torch.float).reshape(1,-1)
            cjps_2=torch.tensor(motion_trajectories[i][1]).type(torch.float).reshape(1,-1)
            joint_ps=torch.cat([cjps_1,cjps_2])
            scene=unique_scenes[obstructed_examples[i]]

            t0=time.time()
            _, success, distances,current_dist,jps_path,grad_dirs,pc=path_adjust(scene,joint_ps,dist_func,motion_scale=motion_scale,dist_threshold=dist_threshold,scale_adjust=scale_factor,max_tries=10)
            example_time=time.time()-t0
            times[method].append(example_time)

            scene_meshes,_=util.create_meshes(scene)
            scene_objs=[]

            for j in range(len(scene_meshes)):

                mesh=scene_meshes[j]

                geom=trimesh.collision.mesh_to_BVH(mesh)
                obj=fcl.CollisionObject(geom)
                scene_objs.append(obj)

            num_true_collisions=0    
    
            test_poses,test_meshes,test_transforms=util.generate_poses_from_jps(robot,jps_path,active_link_names,active_joint_names,sizes)

            for link_meshes in test_meshes:
                true_min_dist_pose=np.inf
                for link_mesh in link_meshes:
            
                    geom=trimesh.collision.mesh_to_BVH(link_mesh)
                    obj=fcl.CollisionObject(geom)
                    md,_,point2,_,_=util.calculate_distance(scene_objs,obj,penetration_depth=True)
                    true_min_dist_pose=np.min([true_min_dist_pose,md])
                if true_min_dist_pose<0:
                    num_true_collisions+=1
            
            collision=np.min([num_true_collisions,1])
            was_successful= success and not collision 
            successes[method]=was_successful

    print("percentage of trimesh successes", np.array(successes["tm"]).sum()/num_examples*100)
    print("average time for trimesh", np.array(times["tm"]).mean(), "+/-",np.array(times["tm"]).std())

    print("percentage of network successes", np.array(successes["net"]).sum()/num_examples*100)
    print("average time for network", np.array(times["net"]).mean(), "+/-",np.array(times["tm"]).std())

    print("percentage of successes without path adjustment", np.array(successes["net"]).sum()/num_examples*100)
    print("average time without path adjustment", np.array(times["net"]).mean(), "+/-",np.array(times["tm"]).std())

