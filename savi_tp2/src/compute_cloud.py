#!/usr/bin/env python3
import open3d as o3d
import rospy
from pointCloud_Transform import cloud2array
# ---------------------------------------------------
#   Load file (for starters is only for testing - might be included in the loop to change over time)
# ---------------------------------------------------
filename = '../../../rgbd-scenes-v2/pcd/01.pcd'
original_pcd = o3d.io.read_point_cloud(filename)
# ---------------------------------------------------
#   Downscale cloud
# ---------------------------------------------------
pcd_ds = original_pcd.voxel_down_sample(voxel_size=0.01)
# ---------------------------------------------------
#   Get table center (original frame), table limits and frame rotations
# ---------------------------------------------------
