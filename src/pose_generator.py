#!/usr/bin/python3

import os
import sys
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R

import torch
from graspnetAPI import GraspGroup

import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from graspnet_baseline_ros.msg import GraspGripperArg


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=os.path.join(ROOT_DIR, 'checkpoints/checkpoint-rs.tar'), help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs, unknown = parser.parse_known_args()

bridge = CvBridge()


class GraspNetROS:
    def __init__(self):
        self.net = self.get_net()
        self.color_image = None
        self.depth_image = None
        self.grasp_start = False
        self.color_sub = Subscriber("/d435_camera/color/image_raw", RosImage)
        # self.depth_sub = Subscriber("/d435_camera/aligned_depth_to_color/image_raw", RosImage)
        self.depth_sub = Subscriber("/d435_camera/depth/image_raw", RosImage)
        self.ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=1, slop=0.05)
        self.ats.registerCallback(self.image_callback)
        self.trigger_sub = rospy.Subscriber("/grasp_start", Bool, self.process_data)
        self.trigger_pub = rospy.Publisher("/grasp_start", Bool, queue_size=1)
        self.pose_pub = rospy.Publisher("/grasp_pose_stamped", PoseStamped, queue_size=1)
        self.gripper_arg_pub = rospy.Publisher("/grasp_gripper_arg_stamped", GraspGripperArg, queue_size=1)

    def image_callback(self, color_msg, depth_msg):
        try:
            self.color_image = bridge.imgmsg_to_cv2(color_msg, "rgb8")
            self.depth_image = bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            print(e)

    def get_net(self):
        net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def process_data(self, msg):
        self.grasp_start = msg.data
        if self.grasp_start and self.color_image is not None and self.depth_image is not None:
            rospy.loginfo("Received True on /grasp_start topic, starting grasp detection.")

            color = np.array(self.color_image, dtype=np.float32) / 255.0
            depth = np.array(self.depth_image)
            workspace_mask = np.ones_like(depth, dtype=bool)  # Assuming whole image is workspace for simplicity
            intrinsic = np.array([[912.135877,          0, 631.980233],
                                  [         0, 913.634983, 357.925331],
                                  [         0,          0,          1]])
            factor_depth = 1000.0

            camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

            mask = (workspace_mask & (depth > 0))
            cloud_masked = cloud[mask]
            color_masked = color[mask]

            if len(cloud_masked) >= cfgs.num_point:
                idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]
            color_sampled = color_masked[idxs]

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
            cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
            end_points = dict()
            cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cloud_sampled = cloud_sampled.to(device)
            end_points['point_clouds'] = cloud_sampled
            end_points['cloud_colors'] = color_sampled

            gg = self.get_grasps(end_points)
            if cfgs.collision_thresh > 0:
                gg = self.collision_detection(gg, np.array(cloud.points))

            self.publish_grasps(gg)
            self.trigger_pub.publish(Bool(data=False))
            self.grasp_start = False
            rospy.loginfo("Grasp detection script finished, waiting for new message.")


    def get_grasps(self, end_points):
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
        return gg
    
    def vis_grasps(self, gg, cloud):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def publish_grasps(self, gg):
        gg.nms()
        gg.sort_by_score()
        best_grasp = gg[0]  # Get the top grasp

        T = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        rotation_matrix = np.dot(T, best_grasp.rotation_matrix)
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=rospy.Time.now(), frame_id="d435_camera_link")
        pose_msg.pose.position.x = best_grasp.translation[2]
        pose_msg.pose.position.y = -best_grasp.translation[0]
        pose_msg.pose.position.z = -best_grasp.translation[1]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Published PoseStamped to /grasp_pose_stamped")

        gripper_arg_msg = GraspGripperArg()
        gripper_arg_msg.header = Header(stamp=rospy.Time.now(), frame_id="4cx_gripper")
        gripper_arg_msg.depth = best_grasp.depth
        gripper_arg_msg.width = best_grasp.width
        self.gripper_arg_pub.publish(gripper_arg_msg)
        rospy.loginfo("Published GraspGripperArg to /grasp_gripper_arg_stamped")

if __name__ == '__main__':
    rospy.init_node('graspnet_ros', anonymous=True)
    graspnet_ros = GraspNetROS()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")