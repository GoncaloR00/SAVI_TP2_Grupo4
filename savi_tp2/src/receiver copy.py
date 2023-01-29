#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import rospy
from savi_tp2.msg import cloudArray


# ---------------------------------------------------
#   Load file (for starters is only for testing - might be included in the loop to change over time)
# ---------------------------------------------------
filename = '../../../rgbd-scenes-v2/pcd/01.pcd'
original_pcd = o3d.io.read_point_cloud(filename)


# ---------------------------------------------------
#   Topics to publish
# ---------------------------------------------------
topic_ptc = '/tp2_savi/clouds/ptc'
pub_ptc = rospy.Publisher(topic_ptc, cloudArray, queue_size=1)
topic_ptcDS = '/tp2_savi/clouds/ptcDS'
pub_ptcDS = rospy.Publisher(topic_ptcDS, cloudArray, queue_size=1)
frame_rate = 20

# ---------------------------------------------------
#   Downscale cloud
# ---------------------------------------------------
pcd_ds = original_pcd.voxel_down_sample(voxel_size=0.01)

# ---------------------------------------------------
#   Convert clouds to numpy arrays
# ---------------------------------------------------
ptc_points = np.asarray(original_pcd.points)
ptc_colors = np.asarray(original_pcd.colors)
ptcDS_points = np.asarray(pcd_ds.points)
ptcDS_colors = np.asarray(pcd_ds.colors)
# # ---------------------------------------------------
# #   Create messages to send
# # ---------------------------------------------------
# ptc_msg = cloudArray()
# ptcDS_msg = cloudArray()
# ptc_msg.points = ptc_points
# ptc_msg.colors = ptc_colors
# ptcDS_msg.points = ptcDS_points
# ptcDS_msg.colors = ptcDS_colors

# def talker():
#     rospy.init_node('receiver', anonymous=False)
#     rate = rospy.Rate(frame_rate) # 10hz
#     while not rospy.is_shutdown():
#         pub_ptc.publish(ptc_msg)
#         pub_ptcDS.publish(ptcDS_msg)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass