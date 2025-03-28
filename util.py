#!/usr/bin/env python
# coding: utf-8

import torch
import copy
import numpy as np
import trimesh
import fcl
import pyrender
import time

from trimesh import viewer

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from numpy.random import default_rng
from urdfpy import URDF as pURDF


rand=default_rng()


##### DISTANCE CALCULATION AND UTILITY FUNCTIONS ######

def get_robot_info():
    active_link_names=["right_shoulder_fe_link","right_shoulder_aa_link","right_shoulder_ie_link","right_elbow_fe_link","right_wrist_rotation_link","right_wrist_flexion_link"]
    active_joint_names=["right_shoulder_fe","right_shoulder_aa","right_shoulder_ie","right_elbow_fe","right_wrist_rotation","right_wrist_flexion"]

    robot=pURDF.load("./test_arm/robot_obj_combined_hulls.urdf")
    sizes=[]
    active_link_meshes=[]
    for link in active_link_names:
        mesh=trimesh.load("./test_arm/meshes/urdf/obj_combined_hulls/"+link+".obj")
        active_link_meshes.append(mesh)
        sizes.append(mesh.bounding_box.extents)
    return robot, active_link_names, active_joint_names, sizes

def rotation_matrix_from_vectors(vec1, vec2):
    
    if np.array_equal(vec1,vec2):
        return np.eye(3)
    
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def pose_to_obj(pose,obj='box'):
    
    link_position=np.array(pose[0])
    link_rotation=np.array(pose[1])
    link_size=np.array(pose[2])
    rot=R.from_quat(link_rotation)

    

    mat=np.zeros([4,4])
    mat[:3,:3]=rot.as_matrix()
    mat[:3,3]=link_position

    mat[3,3]=1

    if obj=='box':        

        link_cube_mesh=trimesh.creation.box(extents=[link_size[0],link_size[1],link_size[2]],transform=mat) 
        link_obj=fcl.CollisionObject(trimesh.collision.mesh_to_BVH(link_cube_mesh))
        
    elif obj=='L':
        
        nm=np.zeros([4,4])
        nm[0,2]=1
        nm[1,1]=1
        nm[2,0]=-1
        nm[3,3]=1
        new_size=[link_size[0]-0.02,link_size[1],link_size[2]-0.02]

        nm[0,3]=(link_size[0])
        nm[2,3]=(link_size[2])/2-0.03

        link_cube_mesh=trimesh.creation.box(extents=link_size,transform=mat)
        link_cube_mesh_2=trimesh.creation.box(extents=new_size,transform=np.matmul(mat,nm))
        finished_mesh = trimesh.util.concatenate( [link_cube_mesh,link_cube_mesh_2]  )
        link_obj=fcl.CollisionObject(trimesh.collision.mesh_to_BVH(finished_mesh))
        
    elif obj=='capsule':
        
        finished_mesh=trimesh.creation.capsule(radius=0.03,height=0.1)
        finished_mesh.apply_transform(mat)
        link_obj=fcl.CollisionObject(trimesh.collision.mesh_to_BVH(finished_mesh))
        

    return link_obj       

def create_meshes(boxes):
    cube_meshes=[]
    pmeshes=[]
    total_area=0
    for i in range(len(boxes)):
        rot=R.from_quat(boxes[i][3:7]).as_matrix()
        size=boxes[i][7:]
        position=boxes[i][:3]
        
        mat=np.zeros([4,4])
        mat[:3,:3]=rot
        mat[:3,3]=position
        mat[3,3]=1
        cube_mesh=trimesh.creation.box(size,mat)
        total_area+=cube_mesh.area
        cube_meshes.append(cube_mesh)
    return cube_meshes,total_area

def create_L_mesh(pose):
    
    rot=R.from_quat(pose[3:7]).as_matrix()
    link_size=pose[7:]
    position=pose[:3]
    
    mat=np.zeros([4,4])
    mat[:3,:3]=rot
    mat[:3,3]=position
    mat[3,3]=1
    
    nm=np.zeros([4,4])
    nm[0,2]=1
    nm[1,1]=1
    nm[2,0]=-1
    nm[3,3]=1
    new_size=[link_size[0]-0.02,link_size[1],link_size[2]-0.02]

    nm[0,3]=(link_size[0])
    nm[2,3]=(link_size[2])/2-0.03

    link_cube_mesh=trimesh.creation.box(extents=link_size,transform=mat)
    link_cube_mesh_2=trimesh.creation.box(extents=new_size,transform=np.matmul(mat,nm))
    finished_mesh = trimesh.util.concatenate( [link_cube_mesh,link_cube_mesh_2]  )

    return finished_mesh
    
    
    
def create_pc(meshes,num_points,total_area):
    total_points=0
    pc=[]
    for cube_mesh in meshes:
        num_points_on_cube=np.floor(num_points*cube_mesh.area/total_area)
        total_points+=num_points_on_cube
        for point in trimesh.sample.sample_surface(cube_mesh,int(num_points_on_cube))[0].tolist():
            pc.append(point)
    for m in range(int(num_points)-int(total_points)):
        pc.append(trimesh.sample.sample_surface(meshes[np.random.randint(0,len(meshes))],1)[0].tolist()[0])
    return torch.tensor(pc).type(torch.float)



def random_joint_position(joint_list):

    joint_position=[]
    for joint in joint_list:
        joint_position.append(joint.limit.lower+rand.random()*(joint.limit.upper-joint.limit.lower))
    return joint_position
def random_rot():
    
    quat=[np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()]
    quat_norm=np.linalg.norm(quat)
    
    return quat/quat_norm
    
def calculate_distance(scene_boxes,link,penetration_depth=False):
    
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(scene_boxes)
    manager.setup()
    drequest=fcl.DistanceRequest(enable_nearest_points=True)
    data=fcl.DistanceData(request=drequest)       
    manager.distance(link,data,fcl.defaultDistanceCallback)
    
    min_distance=data.result.min_distance
    
    if min_distance==0 and penetration_depth:
        req = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
        rdata = fcl.CollisionData(request = req)

        manager1 = fcl.DynamicAABBTreeCollisionManager()
        manager1.registerObjects(scene_boxes)
        manager1.setup()

        manager1.collide(link, rdata, fcl.defaultCollisionCallback)
        
        boxes_in_contact=[]
        for c in rdata.result.contacts:
            if c.o1 not in boxes_in_contact:
                boxes_in_contact.append(c.o1)
        
        box_penetration_dict={}
        for box in boxes_in_contact:
            box_penetration_dict[box]=[]

        for c in rdata.result.contacts:
            box_penetration_dict[c.o1].append(c.penetration_depth)
    
        minimum_depths=[]
        for box in boxes_in_contact:
            minimum_depths.append(np.min(np.array(box_penetration_dict[box])))
    
        min_distance=-np.max(np.array(minimum_depths))
    return min_distance,data.result.nearest_points[0],data.result.nearest_points[1],data.result.o1,data.result.o2
    
def generate_poses_from_jps(robot,joint_positions,active_link_names,active_joint_names,sizes):
    poses=[]
    meshes=[]
    tfks=[]
    for jps in joint_positions:
        cfg={}
        for i in range(len(jps)):
            cfg[active_joint_names[i]]=jps[i]

        tfk=robot.collision_trimesh_fk(cfg,active_link_names)
        test_meshes=[]    
        before_transform_test_meshes=[]
        k=0
        link_poses=[]
        link_meshes=[]
        link_transforms=[]
        for link in tfk:
            transformation=tfk[link]
            link_transforms.append(transformation)
            mesh=copy.deepcopy(link)
            link_meshes.append(mesh.apply_transform(transformation))
            position=transformation[:3,3]
            rot=R.from_matrix(transformation[:3,:3]).as_quat()
            size=sizes[k]
            link_poses.append(np.concatenate((position,rot,size)))
            k+=1
        link_poses=np.array(link_poses)
        poses.append(link_poses)
        meshes.append(link_meshes)
        tfks.append(link_transforms)
    poses=np.array(poses)
    return poses, meshes, tfks
    
    
    
def animate_traj(scene,box_traj,initial_meshes,grad_traj,box_sizes,fps=24,obj_type='box',show_frame=False,pc=None):
    
    show_grad=False
    if len(grad_traj)>0:
        show_grad=True

    print("showing grad", show_grad)
    
    scene_meshes,_=create_meshes(scene)
    
    total_positions=len(box_traj)

    initial_rmats=box_traj[0]
    box_reset_mats=[]
    for i_rmat in initial_rmats:
        box_reset_mats.append(np.linalg.inv(i_rmat))

    pscene=pyrender.Scene()

    node_map={}

    if show_grad:

        initial_amat=grad_traj[0]
        arrow=trimesh.primitives.Cylinder(radius=0.005,height=0.05,transform=initial_amat)
        arrow_reset_mat=np.linalg.inv(initial_amat)
        amesh=pyrender.Mesh.from_trimesh(arrow)
        
        anode=pscene.add(amesh)
        node_map[arrow]=anode      
        
        
    if show_frame:
        v1=np.array([1,0,0])
        v2=np.array([0,1,0])
        v3=np.array([0,0,1])

        cylin1=trimesh.creation.cylinder(radius=0.005,height=0.2)
        cylin2=trimesh.creation.cylinder(radius=0.005,height=0.2)
        cylin3=trimesh.creation.cylinder(radius=0.005,height=0.2)

        cmat1=np.eye(4)
        cmat2=np.eye(4)
        cmat3=np.eye(4)


        rot1=rotation_matrix_from_vectors(v1,v2)
        rot2=rotation_matrix_from_vectors(v1,v3)
        rot3=rotation_matrix_from_vectors(v2,v3)
        cmat1[:3,:3]=rot3
        cmat2[:3,:3]=rot2

        cylin1.apply_transform(cmat1)
        cylin2.apply_transform(cmat2)
        cylin3.apply_transform(cmat3)

        frames=[]
        fnodes=[]
        frame=trimesh.util.concatenate([cylin1,cylin2,cylin3])
        for i_rmat in initial_rmats:
            frames.append(copy.deepcopy(frame).apply_transform(i_rmat))
            fnodes.append(pscene.add(pyrender.Mesh.from_trimesh(frames[-1],smooth=False)))
            nodemap[frames[-1]]=fnodes[-1]
            
    for m in initial_meshes:
        m.visual.face_colors=[0,50,50,100]
        mesh=pyrender.Mesh.from_trimesh(m,smooth=False)
        node=pscene.add(copy.deepcopy(mesh))
        node_map[m]=node
    for s in scene_meshes:

        mesh=pyrender.Mesh.from_trimesh(s,smooth=False)
        node=pscene.add(copy.deepcopy(mesh))
        node_map[s]=node
        
    if pc!=None:
        mesh=pyrender.Mesh.from_points(pc)
        node=pscene.add(copy.deepcopy(mesh))
  
    v = pyrender.Viewer(pscene, run_in_thread=True,use_raymond_lighting=True)
    
    time.sleep(1.0/fps)

    iteration=1
    while v.is_active:
        current_pos=iteration%total_positions
        
        rmats=box_traj[current_pos]
        
        v.render_lock.acquire()
        
        for i in range(len(rmats)):
            node_map[initial_meshes[i]].matrix=np.matmul(rmats[i],box_reset_mats[i])
            
            if show_frame:
                node_map[frame[i]].matrix=np.matmul(rmats[i],box_reset_mats[i])
        
        if show_grad:
            amat=grad_traj[current_pos]
            node_map[arrow].matrix=np.matmul(amat,arrow_reset_mat)

        v.render_lock.release()

        time.sleep(1.0/fps)
        iteration+=1

        
def create_link_and_grad_rot_matricies(poses,grad_dirs,nearest_points):
    rmats=[]
    amats=[]
    
    num_examples=len(poses)
    
    for i in range(num_examples):
        pose=poses[i]
        np_pose=pose.detach().cpu().numpy()
        rot=R.from_quat(np_pose[3:7]).as_matrix()
        position=np_pose[:3]
        
        rmat=np.eye(4)
        rmat[:3,:3]=rot
        rmat[:3,3]=position
        rmats.append(rmat)
        
        if len(grad_dirs)==len(poses):
        
            nearest_point=nearest_points[i]
            grad_dir=grad_dirs[i]
        
            v1=np.array([0,0,1])
            v2=grad_dir/np.linalg.norm(grad_dir)
            rot=rotation_matrix_from_vectors(v1,v2)

            mat=np.eye(4)
            mat[:3,:3]=rot
        
            tmat=np.eye(4)
            tmat[:3,3]=[0,0,-0.1/2]
    
            nmat=np.eye(4)
            nmat[:3,3]=nearest_point
            amat=np.matmul(nmat,np.matmul(mat,tmat))
            
            amats.append(amat)
    return rmats,amats
        
  
def gen_local_pc(robot,active_link_names,active_joint_names,sizes,jps,pc):

    test_poses,_,_=generate_poses_from_jps(robot,[jps],active_link_names,active_joint_names,sizes)
    
    test_poses=test_poses.reshape(6,-1)
    
    nearness=0.005
    nogpc=pc[torch.where(((torch.abs(pc-test_poses[0,:3])<nearness)).sum(dim=1)==3)[0].detach().cpu().numpy(),:]
    for i in range(1,test_poses.shape[0]):
        nogpc=torch.cat([nogpc,pc[torch.where(((torch.abs(pc-test_poses[i,:3])<nearness)).sum(dim=1)==3)[0].detach().cpu().numpy(),:]],dim=0)

    while nogpc.shape[0]<600:
        nearness+=0.005
        nogpc=pc[torch.where((torch.abs(pc-test_poses[0,:3])<nearness).sum(dim=1)==3)[0].detach().cpu().numpy(),:]
        for i in range(1,test_poses.shape[0]):
            nogpc=torch.cat([nogpc,pc[torch.where(((torch.abs(pc-test_poses[i,:3])<nearness)).sum(dim=1)==3)[0].detach().cpu().numpy(),:]],dim=0)
    
    if nogpc.shape[0]>600:
        nogpc=nogpc[torch.randperm(nogpc.size()[0]),:]

        nogpc=nogpc[:600,:]

    return nogpc



def rot_dir_from_point(p,rp,theta):
    
    
    def calc_dis(p,rp,theta):
        f1=torch.cos(theta[1])*torch.cos(theta[2])
        f2=torch.sin(theta[0])*torch.sin(theta[1])*torch.cos(theta[2]) - torch.sin(theta[2])*torch.cos(theta[0])
        f3=torch.sin(theta[1])*torch.cos(theta[0])*torch.cos(theta[2]) + torch.sin(theta[0])*torch.sin(theta[2])
        f4=torch.sin(theta[2])*torch.cos(theta[1])
        f5=torch.sin(theta[0])*torch.sin(theta[1])*torch.sin(theta[2])+ torch.cos(theta[0])*torch.cos(theta[2])
        f6=torch.sin(theta[1])*torch.sin(theta[2])*torch.cos(theta[0]) - torch.sin(theta[0])*torch.cos(theta[2])
        f7=-torch.sin(theta[1])
        f8=torch.sin(theta[0])*torch.cos(theta[1])
        f9=torch.cos(theta[0])*torch.cos(theta[1])

        x_co=torch.pow(((rp[0]*f1+rp[1]*f2+rp[2]*f3)-p[0]),2)
        y_co=torch.pow(((rp[0]*f4+rp[1]*f5+rp[2]*f6)-p[1]),2)
        z_co=torch.pow(((rp[0]*f7+rp[1]*f8+rp[2]*f9)-p[2]),2)

        dis=torch.sqrt(x_co+y_co+z_co)    
        

        return dis
    theta_t=torch.tensor(theta,requires_grad=True)
    #theta=torch.tensor([0.,0.,0.],requires_grad=True)
    dis=calc_dis(p,rp,theta_t)

    dis.backward()
    
    return theta_t.grad
    

def create_grad_rot_matricies(grad_dirs,nearest_points):
    rmats=[]
    amats=[]
    
    num_examples=len(grad_dirs)
    
    for i in range(num_examples):
        
        nearest_point=nearest_points[i]
        grad_dir=grad_dirs[i]
        
        v1=np.array([0,0,1])
        v2=grad_dir/np.linalg.norm(grad_dir)
        rot=rotation_matrix_from_vectors(v1,v2)

        mat=np.eye(4)
        mat[:3,:3]=rot
        
        tmat=np.eye(4)
        tmat[:3,3]=[0,0,0.05/2]
    
        nmat=np.eye(4)
        nmat[:3,3]=nearest_point
        amat=np.matmul(nmat,np.matmul(mat,tmat))
            
        amats.append(amat)
    return amats
