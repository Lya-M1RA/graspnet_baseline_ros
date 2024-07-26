# graspnet_baseline_ros
A ROS wrapper of [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline)

## Requirements
The code has been modified for **ROS Noetic** and the latest PyTorch versions.
- Python 3.8
- PyTorch>=1.8
- Open3d
- TensorBoard
- NumPy==1.23.5
- SciPy
- Pillow
- tqdm

## Installation
Create your ROS workspace and get the code.
```bash
mkdir -p graspnet_ws/src
cd graspnet_ws/src
git clone https://github.com/Lya-M1RA/graspnet_baseline_ros.git
cd graspnet_baseline_ros
```

Install packages via Pip.
```bash
pip install -r requirements.txt
```

If you are using virtual environment. Modify the first line of `pose_generator.py` to specify your interpreter path. Then install ROS required packages.
```bash
pip install rospkg rospy catkin_tools empy==3.3.4
```

Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd src/pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd src/knn
python setup.py install
```
Install graspnetAPI for evaluation (code adapted from [graspnetAPI](https://github.com/graspnet/graspnetAPI))..
```bash
cd src/graspnetAPI
pip install .
```

The pretrained weights can be downloaded from:

- `checkpoint-rs.tar`
[[Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing)]
[[Baidu Pan](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A)]
- `checkpoint-kn.tar`
[[Google Drive](https://drive.google.com/file/d/1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX/view?usp=sharing)]
[[Baidu Pan](https://pan.baidu.com/s/1QpYzzyID-aG5CgHjPFNB9g)]

`checkpoint-rs.tar` and `checkpoint-kn.tar` are trained using RealSense data and Kinect data respectively. Put the pretrained weights in `src/checkpoints`.  

`--checkpoint_path` in `pose_generator.py` should be specified according to your settings (make sure you have downloaded the pretrained weights, we recommend the realsense model since it might transfer better).

Now you can use catkin to build the package.

## Usage
Launch the node.
```bash
roslaunch graspnet_baseline_ros graspnet_pose_generator.launch
```
Some parts of `pose_generator.py` should be modified before use to fit your configuration.

The node subscribes the color and depth images topic of the RGBD camera, and would be triggered once if receive a `True` on `/grasp_start` topic. The generated grasp pose and the gripper width and depth are published to `/grasp_pose_stamped` and `/grasp_gripper_arg_stamped`.
