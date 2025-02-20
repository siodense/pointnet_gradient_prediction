# pointnet_gradient_prediction

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

To train a new network first use python pc_gen.py to generate point_clouds from the scene_descriptions file, then simply use python train.py to train a new model.

To test the existing trained model use python arm_test_time.py with the parameter visualize to visualize a specific method/example and test to compare methods on the entire dataset. The parameters are -m,-e,-s,-d,-f,-t for the method,
example number, motion scale, distance threshold, scale factor, and maximum attempts respectively. The first two are only relevant for the visualize task as test will run every method on every example and report the success rate and 
average time taken for a successful example

