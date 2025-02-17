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
import sys
import getopt


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

def trimesh_func(pc,jps,num_tm_points=1):
    
    poses,meshes,_=util.generate_poses_from_jps(robot,[jps],active_link_names,active_joint_names,sizes)
    
    closest_points_in_pc=[]
    closest_points_on_meshes=[]
    trimesh_mds=[]
    num_links=len(meshes[0])
    
    #Ignore the first link, it has limited movement and building a chain from a single link causes errors. 
    for i in range(1,num_links):
        
        mesh=meshes[0][i]
        
        closest_points,dist,_=trimesh.proximity.closest_point(mesh,pc)
                
        link_mds=[]
        link_closest_points=[]
        link_closest_pc_points=[]
        
        
        for j in range(num_tm_points):
            
            closest_point_ind=np.argmin(dist)
    
            link_closest_points.append(closest_points[closest_point_ind])
    
            link_closest_pc_points.append(pc[closest_point_ind])
                
            link_mds.append(dist[closest_point_ind])
            dist=np.concatenate((dist[:closest_point_ind],dist[closest_point_ind+1:]))
            
        closest_points_in_pc.append(link_closest_pc_points)
        closest_points_on_meshes.append(link_closest_points)
        trimesh_mds.append(link_mds)
        
    closest_points_in_pc=np.array(closest_points_in_pc)
    closest_points_on_meshes=np.array(closest_points_on_meshes)
    trimesh_mds=np.array(trimesh_mds)
    
    mds=[]
    gds=[]
    for i in range(num_tm_points):            
            
        link_ind=np.argmin(trimesh_mds[:,i])

        position=poses[0][link_ind+1][:3]
        closest_point_in_pc=closest_points_in_pc[link_ind][i]
        closest_point_on_mesh_tm=(closest_points_on_meshes[link_ind][i])
        current_workspace_gd=torch.tensor(closest_point_on_mesh_tm)-closest_point_in_pc
        current_workspace_gd=current_workspace_gd.type(torch.float)

        chain=chains[link_ind]
        offset=closest_point_on_mesh_tm-position
                
        J=chain.jacobian(jps[:link_ind+2],locations=offset)
  
        J=J[0]
        J_inv=J.transpose(0,1)
        #Can calculate a rotation angle to move the point away with util.rot_dir_from_point but doesn't seem
        #to make much difference in the success
        rot_dir=torch.zeros(3)
        padded_workspace_gd=torch.cat((current_workspace_gd,rot_dir))

        gd=torch.zeros([6])
        gd[:link_ind+2]=torch.tensor(torch.matmul(J_inv,padded_workspace_gd))
            
        mds.append(trimesh_mds[link_ind][i])
        gds.append(gd.detach().cpu().numpy())
            
        new_workspace_gd=torch.matmul(J,gd[:link_ind+2])

    for i in range(len(mds)):
        gds[i]=gds[i]*mds[0]/mds[i]
    gd=-1*np.array(gds).sum(axis=0)
    md=mds[0]
    
                    
    return md,gd,-1,[],[]
    
def combined_dist_func(network,pc,jps,normalize=False,use="network",num_tm_points=1):
    

    #Using Trimesh to calculte the minimum distance and the nearest point on the mesh in order to visualize
    #The gradient
    poses,meshes,_=util.generate_poses_from_jps(robot,[jps],active_link_names,active_joint_names,sizes)
    
    closest_points_in_pc=[]
    closest_points_on_meshes=[]
    trimesh_mds=[]
    num_links=len(meshes[0])
    
    for i in range(1,num_links):
        
        mesh=meshes[0][i]
        
        closest_points,dist,_=trimesh.proximity.closest_point(mesh,pc)
                
        link_mds=[]
        link_closest_points=[]
        link_closest_pc_points=[]
            
        for j in range(num_tm_points):
    
            closest_point_ind=np.argmin(dist)
    
            link_closest_points.append(closest_points[closest_point_ind])
    
            link_closest_pc_points.append(pc[closest_point_ind])
                
            link_mds.append(dist[closest_point_ind])
            dist=np.concatenate((dist[:closest_point_ind],dist[closest_point_ind+1:]))

        closest_points_in_pc.append(torch.stack(link_closest_pc_points))
        closest_points_on_meshes.append(np.stack(link_closest_points))
        trimesh_mds.append(link_mds)

    closest_points_in_pc=torch.stack(closest_points_in_pc)
    closest_points_on_meshes=np.stack(closest_points_on_meshes)
    trimesh_mds=np.array(trimesh_mds)
    
    #Calculates the C-space gradient and minimum distance using the network
    if use=="network":
        
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

        #Ignore first link, it has limited mobility
        min_dist_pred=min_dist_pred[0][1:6]/100

        overall_min_dist=min_dist_pred.min()
        
        closest_link=min_dist_pred.argmin()

        overall_min_dist.backward()

        grad_dir=jps.grad


        md=overall_min_dist.detach().cpu().numpy()

        gd=grad_dir.detach().cpu().numpy()
                
        closest_point_in_pc=closest_points_in_pc[closest_link][0]        
        closest_point=closest_points_on_meshes[closest_link][0]
        
        workspace_gd=torch.tensor(closest_point)-closest_point_in_pc
        
    #If we aren't using the network compute the workspace gradient from trimesh or are not computing the gradient
    else:

        mds=[]
        gds=[]
        
        #can't build a chain with a single link, so we look for the closest points on the other links. The first link
        #can't move much anyways, for some reason this isn't a problem in the speed test code, have to look into it more
        #trimesh_mds=trimesh_mds[1:,:]
        
        for i in range(num_tm_points):
            
            
            link_ind=np.argmin(trimesh_mds[:,i])
            


            position=poses[0][link_ind+1][:3]
            closest_point_in_pc=closest_points_in_pc[link_ind][i]
            closest_point_on_mesh_tm=(closest_points_on_meshes[link_ind][i])
            current_workspace_gd=torch.tensor(closest_point_on_mesh_tm)-closest_point_in_pc
            current_workspace_gd=current_workspace_gd.type(torch.float)
            if i==0:
                closest_point=closest_point_on_mesh_tm
                closest_link=link_ind+1
                
        
        #Uses trimesh to calculate the gradient to use in the path adjustment
            if use=="trimesh":

                chain=chains[link_ind]
                th=jps[:link_ind+2]
                th.requires_grad=True
                tg=chain.forward_kinematics(th)
                m=tg.get_matrix()
                m_inv=torch.linalg.inv(m[0])
            
                appended_cpomt=torch.cat((torch.tensor(closest_point_on_mesh_tm).type(torch.float),torch.tensor([1.])))
            
                pre_transform_point=torch.matmul(m_inv,appended_cpomt).detach().clone()
                transform_point=torch.matmul(m,pre_transform_point)
                distance=(transform_point[0][:3]-closest_point_in_pc).norm()
                distance.backward()
                gd=th.grad
                offset=closest_point_on_mesh_tm-position
                
                J=chain.jacobian(jps[:link_ind+2],locations=offset)
                
                J=J[0]
                J_inv=J.transpose(0,1)
                rot_dir=torch.zeros(3)
                padded_workspace_gd=torch.cat((current_workspace_gd,rot_dir))

                gd=torch.zeros([6])
                gd[:link_ind+2]=torch.matmul(J_inv,padded_workspace_gd)
            
                mds.append(trimesh_mds[link_ind][i])
                gds.append(gd.detach().cpu().numpy())
            
                #new_workspace_gd=torch.matmul(J,gd[:link_ind+2])[:3]
            
                if i==0:
                    new_workspace_gd=np.matmul(J,gd[:link_ind+2])[:3]
                    workspace_gd=new_workspace_gd
            
        #If no method is specified assume no perterbation is intended, this will produce the path as is without 
        #attempting to avoid collisions
            else:

                md=np.inf
                gd=np.array([])
                workspace_gd=current_workspace_gd
                
                return md,gd,closest_link,closest_point,workspace_gd

        for i in range(len(mds)):
            gds[i]=gds[i]*mds[0]/mds[i]
        gd=-1*np.array(gds).sum(axis=0)
        md=mds[0]

    return md,gd,closest_link,closest_point,workspace_gd

    
def null_func(pc,jps):
    return np.inf, np.array([])





"""
Given a path in a scene and a distance function, try to move the robot arm along the path, if the distance function
detects that the robot arm is too close to an object in the scene adjust the trajectory using the gradient returned
by the distance function in order to avoid the object. Also record the nearest points between the robot and the scene, this expects you
To use combined_dist_func which computes this information as well, this will be slower but allow you to visualize 
The gradient direction

"""

def path_adjust(scene,jps,dist_func,motion_scale=0.02,dist_threshold=0.02,scale_adjust=1,max_tries=200):
    
    distances=[]
    path_poses=[]
    grad_directions=[]
    closest_links=[]
    closest_points=[]
    workspace_grad_directions=[]
    num_collisions=0
    
    current_jps=jps[0]
    path_poses.append(current_jps)
    
    scene_meshes,ta=util.create_meshes(scene)
    pc=util.create_pc(scene_meshes,20000,ta)

    total_checkpoints=jps.shape[0]
    #print("jps",jps)
    md,gd,closest_link,closest_point,workspace_grad=get_grad(pc,dist_func,current_jps)
    
    grad_directions.append(gd)
    closest_links.append(closest_link)
    closest_points.append(closest_point)
    workspace_grad_directions.append(workspace_grad)
    distances.append(md)
    next_checkpoint=1
    next_jps=jps[next_checkpoint]
    next_checkpoint_dist=check_goal(current_jps,next_jps)
            
    num_tries=0

    while num_tries<max_tries and next_checkpoint<total_checkpoints:
        print("step: ",num_tries)
        print("distance to checkpoint: ",next_checkpoint_dist)
        print("distance to environment: ",md)
        #print("total_checkpoints",total_checkpoints)
        #print("num tries", num_tries)
        #print("max tries", max_tries)

        num_tries+=1    
        
        direction_vector=next_jps-current_jps
                
        norm_dv=direction_vector/np.linalg.norm(direction_vector)
        
        if md<=0:
            num_collisions+=1 
            
        if gd.any():  
            
            gd=torch.from_numpy(gd)
            norm_gd=gd/torch.linalg.norm(gd)  
            
            #Check if the vector from the gradient is pushing in the direction opposite of the vector to the goal
            #if so add a new vector mutually orthogonal to both, this helps prevent the arm getting stuck
            if  torch.dot(norm_dv,norm_gd)<-0.8:
                                    
                orth_vec=torch.tensor([1.,1.,1.,1.,1.,1.])
                orth_vec-=orth_vec.dot(norm_dv)*norm_dv
                orth_vec/=torch.linalg.norm(orth_vec)

                norm_dv+=orth_vec
                
            if md>dist_threshold:
                scale=1/pow((1+10*(md-dist_threshold)),2)
                norm_gd=norm_gd*scale
                    
            else:
                norm_gd=norm_gd*scale_adjust
                
            direction_vector=norm_dv+norm_gd

            norm_dv=direction_vector/np.linalg.norm(direction_vector)

        norm_dv=norm_dv*motion_scale
        
        current_jps=current_jps+norm_dv
        
        path_poses.append(current_jps)

                   
        next_checkpoint_dist=check_goal(current_jps,next_jps)
                   
        md,gd,closest_link,closest_point,workspace_grad=get_grad(pc,dist_func,current_jps)
        grad_directions.append(gd)
        distances.append(md)
        closest_links.append(closest_link)
        closest_points.append(closest_point)
        workspace_grad_directions.append(workspace_grad)
          
        #print("current jps",current_jps)
        #print("next jps", next_jps)
        #print("next checkpoint dist",next_checkpoint_dist)
        if next_checkpoint_dist<0.02:
            next_checkpoint+=1
           
            if next_checkpoint>=total_checkpoints:
                goal_reached = True
                return num_collisions, goal_reached, distances,next_checkpoint_dist,path_poses,grad_directions,closest_links,closest_points,workspace_grad_directions,pc

            
            next_jps=jps[next_checkpoint]
                
    goal_reached= False

    return num_collisions, goal_reached, distances,next_checkpoint_dist,path_poses,grad_directions,closest_links,closest_points,workspace_grad_directions,pc


def get_grad(pc,dist_func,jps):
    
    nogpc=util.gen_local_pc(robot,active_link_names,active_joint_names,sizes,jps,pc)
            
    md,gd,closest_link,closest_point,workspace_grad=dist_func(nogpc,jps)
            
    return md,gd,closest_link,closest_point,workspace_grad

def check_goal(jps_i,jps_f):
    
    test_pose,_,_=util.generate_poses_from_jps(robot,[jps_i,jps_f],active_link_names,active_joint_names,sizes)
    
    current_ee_position=test_pose[0,-1,:3]
    final_ee_position=test_pose[1,-1,:3]
    
    
    return np.linalg.norm(final_ee_position-current_ee_position)



if __name__=="__main__":

    robot, active_link_names, active_joint_names, sizes=util.get_robot_info()
    
    scene_descriptions=np.load("scene_descriptions.npy",allow_pickle=True)
    unique_scenes=[scene_descriptions[i*10] for i in range(1000)]
    motion_trajectories=np.load("final_arm_test_motion_examples.npy", allow_pickle=True)
    obstructed_examples=np.load("final_arm_test_obstructed_indicies.npy",allow_pickle=True)

    chains=[]

    for i in range(1,len(active_link_names)):
        chains.append(pk.build_serial_chain_from_urdf(open("./lsa1.1_new/robot_obj_combined_hulls.urdf").read(), active_link_names[i],"right_shoulder_fe_link"))


    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=PN(12,in_channel=9)
    model.load_state_dict(torch.load("new_best_model_arm_jps_sc_2.pth",map_location=device))
    model=model.eval()
    new_model=torch.compile(model)

    model=model.to(device)

    #Default values for when no input is given
    motion_scale=0.01
    dist_threshold=0.02
    scale_factor=1000
    max_tries=600
    
    method="network"
    example=10
    
    #Parse arguments to either visualize an example or run comparative tests

    if len(sys.argv)<2 or sys.argv[1] not in ["visualize","test"]:
        print("Useage: visualize -method <method name> -example <example number> or test. motion_scale, dist_threshold, and scale_factor are optional parameters for both visualize and test")
        exit()

    task=sys.argv[1]    
    argumentList = sys.argv[2:]


    options = "m:,e:,s:,d:,f:,t:"

    long_options = ["method=", "example=", "motion_scale=","dist_threshold=","scale_factor=","max_tries="]

    try:

        arguments, values = getopt.getopt(argumentList, options, long_options)


        for currentArgument, currentValue in arguments:
            print(currentArgument,currentValue)
            if currentArgument in ("-m", "--method"):
                method=currentValue
            elif currentArgument in ("-e", "--example"):
                example=int(currentValue)
            elif currentArgument in ("-s", "--motion_scale"):
                motion_scale=float(currentValue)
            elif currentArgument in ("-d", "--dist_threshold"):
                dist_threshold=float(currentValue)
            elif currentArgument in ("-f", "--scale_factor"):
                scale_factor=float(currentValue)
            elif currentArgument in ("-t", "--max_tries"):
                max_tries=float(currentValue)

    except getopt.error as err:
        print (str(err))
    
    
    if task=="test":
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
                _, success, distances,current_dist,jps_path,grad_dirs,pc=path_adjust(scene,joint_ps,dist_func,motion_scale=motion_scale,dist_threshold=dist_threshold,scale_adjust=scale_factor,max_tries=max_tries)
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
        
    elif task=="visualize":
        print("method",method)
        dist_func = lambda x,y:combined_dist_func(model,x,y, use=method,num_tm_points=1)

        cjps_1=torch.tensor(motion_trajectories[example][0]).type(torch.float).reshape(1,-1)
        cjps_2=torch.tensor(motion_trajectories[example][1]).type(torch.float).reshape(1,-1)
        joint_ps=torch.cat([cjps_1,cjps_2])
        scene=unique_scenes[obstructed_examples[example]]
        _,_,_,_,jps_path,grad_dirs,closest_links,closest_points,workspace_grad_directions,pc=path_adjust(scene,joint_ps,dist_func,motion_scale=motion_scale,dist_threshold=dist_threshold,scale_adjust=scale_factor,max_tries=max_tries)

        grm_net=util.create_grad_rot_matricies(workspace_grad_directions,closest_points)

        test_traj_8=util.generate_poses_from_jps(robot,jps_path,active_link_names,active_joint_names,sizes)
        util.animate_traj(scene[:],test_traj_8[2],test_traj_8[1][0],grm_net,sizes,fps=24,obj_type='box',show_frame=False,pc=None)

