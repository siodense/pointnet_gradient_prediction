import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys

from torch.utils.data import TensorDataset, DataLoader, Dataset

from networks import PN_arm as PN

from torch.optim import Adam
from pointnet2_utils import PointNetSetAbstraction

def train(model, optimizer, loader,loss_fn):

    total_loss = 0
    for xyz,out in loader:

        model.train()
        xyz=xyz.permute(0,2,1)
        xyz = xyz.to(device)
        out=out.to(device)*100
        out=out.reshape(-1)
        model_out = model(xyz)[0].reshape(-1)
        loss = loss_fn(model_out, out).type(torch.float)
        optimizer.zero_grad()
        loss.backward()

        total_loss += loss.item()
        optimizer.step()

    return total_loss / (len(loader.dataset))

def eval_loss(model, loader,loss_fn):
    model=model.eval()

    loss = 0

    for xyz,out in loader:
        optimizer.zero_grad()
        xyz = xyz.to(device)
        out=out.to(device)*100
        out=out.reshape(-1)
        with torch.no_grad():
            model_out = model(xyz.permute(0,2,1))[0].reshape(-1)

        itemloss= loss_fn(model_out, out).type(torch.float).squeeze().item()
        loss+=itemloss

    return loss / (len(loader.dataset))
    
class PC_jps_Dataset(Dataset):
    def __init__(self,min_distances,arm_positions,joints,num_points,file_ids, rootdir,filename):
        assert len(file_ids)==arm_positions.shape[0]==min_distances.shape[0]
        self._root_dir = rootdir
        self._total_data = len(file_ids)
        self._filename=filename
        self._arm_positions=arm_positions
        self._ids=file_ids
        self._min_distances=min_distances
        self.num_points=num_points
        self._joint_configs=joints



    def __getitem__(self, idx):

        points=np.loadtxt(self._root_dir+self._filename+str(self._ids[idx])+".txt")

        pc=torch.from_numpy(points)
        pc=pc.type(torch.float)
        arm_positions=copy.deepcopy(self._arm_positions[idx])
        joint_config=copy.deepcopy(self._joint_configs[idx])
        nearness=0.005
        nogpc=pc[torch.where(((torch.abs(pc-arm_positions[0,:3])<nearness)).sum(dim=1)==3)[0].detach().cpu().numpy(),:]
        for i in range(1,arm_positions.shape[0]):
            nogpc=torch.cat([nogpc,pc[torch.where(((torch.abs(pc-arm_positions[i,:3])<nearness)).sum(dim=1)==3)[0].detach().cpu().numpy(),:]],dim=0)

    
        while nogpc.shape[0]<600:
            nearness+=0.005
            nogpc=pc[torch.where((torch.abs(pc-arm_positions[0,:3])<nearness).sum(dim=1)==3)[0].detach().cpu().numpy(),:]
            for i in range(1,arm_positions.shape[0]):
                nogpc=torch.cat([nogpc,pc[torch.where(((torch.abs(pc-arm_positions[i,:3])<nearness)).sum(dim=1)==3)[0].detach().cpu().numpy(),:]],dim=0)
        nogpc=nogpc[torch.randperm(nogpc.shape[0])]

        md=self._min_distances[idx]

        if nogpc.shape[0]>600:
            nogpc=nogpc[torch.randperm(nogpc.size()[0]),:]

            nogpc=nogpc[:600,:]
            
        to_append=joint_config.repeat(nogpc.shape[0],1)

        augmented_pc=torch.cat([nogpc,to_append],dim=1)

        return augmented_pc,md

    def __len__(self):
        return self._total_data

if __name__=="__main__":


    if len(sys.argv)<3:
        print("requires parameters <model_name> <number of epochs>")

    model_name=sys.argv[1]
    epochs=int(sys.argv[2])

    num_examples=100
    batch_size=5
    lr=1e-3

    mds=np.load("python_min_dist_arm.npy",allow_pickle=True)
    poses=np.load("python_box_descripts_arm.npy",allow_pickle=True)
    joints=np.load("python_jps_arm.npy",allow_pickle=True)
    scene_descriptions=np.load("scene_descriptions.npy",allow_pickle=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mds=torch.tensor(mds).type(torch.float)
    joints=torch.tensor(joints).type(torch.float)

    poses=torch.tensor(poses[:,:,:3]).type(torch.float)

    indicies=[i for i in range(num_examples)]
    test_indicies=list(np.random.choice(indicies,size=int(np.floor(num_examples*0.2)),replace=False))
    train_indicies=list(set(indicies)-set(test_indicies))

    train_dataset=PC_jps_Dataset(mds[train_indicies],poses[train_indicies],joints[train_indicies],5000,train_indicies,"./point_clouds_arm/","pc")
    test_dataset=PC_jps_Dataset(mds[test_indicies],poses[test_indicies],joints[test_indicies],5000,test_indicies,"./point_clouds_arm/","pc")

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size)
    
    
    epochs = 300
    bl=np.inf
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PN(12,in_channel=9)
    model=model.to(device)


    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    print("starting")
    for epoch in range(epochs):

        running_loss=train(model,optimizer,train_loader,criterion)
        print("Epoch:", epoch)

        print("Average minibatch loss:", running_loss)
        if epoch %10==0 or epoch >150:

            current_test_loss=eval_loss(model,test_loader,criterion)
            print("Test Loss:", current_test_loss)

            if current_test_loss<bl:
                bl=current_test_loss
                torch.save(model.state_dict(),"./models/" + model_name +".pth")
        if epoch%50==0 and epoch<=200:
            lr=lr/2
            optimizer=torch.optim.Adam(model.parameters(),lr=lr)
