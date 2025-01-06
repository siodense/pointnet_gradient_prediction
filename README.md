# pointnet_gradient_prediction

 e
The following packages are needed:
pytorch
urdfpy
trimesh
pytorch kinematics
pytorch 3d
pyrender
fcl

To run on a specific example and visualize the result use python visualize_examplel.py <method_name> <example_number> where method name is "network", "trimesh", or "none", example number is taken from motion trajectories with
obstructed_examples recording which example from scene_descriptions is to be run on

To compare methods on the whole dataset use python arm_test_time.py, this will run each example with each method and compute the average number of successes and completion time for all methods.
