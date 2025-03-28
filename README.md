# pointnet_gradient_prediction

## Description

Pytorch methods for training a network to predict the minimum distance of a robot to the environment along with the gradient. Contains example problems for avoiding obstacles along a path using the gradient returned from this method
as well as code to test this method against Trimesh point-cloud distance prediction as a baseline.


## Installation
The following packages are needed:
pytorch
urdfpy
trimesh
pytorch kinematics
pytorch 3d
pyrender
fcl


Because certain versions of these packages conflict, an Anaconda environment with the correct versions has been provided with spec-file.txt, use conda install --name myenv --file spec-file.txt to install a working environment
then clone the repository.

## Training
To train a new network first use python pc_gen.py to generate point_clouds from the scene_descriptions file, then simply use python train.py to train a new model.

## Testing
To test the existing trained model use python arm_test_time.py with the parameter visualize to visualize a specific method/example and test to compare methods on the entire dataset. The parameters are -m,-e,-s,-d,-f,-t for the method,
example number, motion scale, distance threshold, scale factor, and maximum attempts respectively. The first two are only relevant for the visualize task as test will run every method on every example and report the success rate and 
average time taken for a successful example

Example: python arm_test_time.py visualize -m trimesh -e 400 to run example 400 using trimesh to calculate distance and gradient

![avoid](https://github.com/user-attachments/assets/4f72befe-f3ee-4365-b44c-33b8292f80b5)


Note pointnet2_utils.py is a modified versoin of the pointnet2_utils.py file from https://github.com/erikwijmans/Pointnet2_PyTorch

