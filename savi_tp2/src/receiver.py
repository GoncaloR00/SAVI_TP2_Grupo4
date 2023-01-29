#!/usr/bin/env python3

import open3d as o3d
import rospy
from pointCloud_Transform import cloud2array
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
ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB = cloud2array(original_pcd)
ptcDS_pointsX, ptcDS_pointsY, ptcDS_pointsZ, ptcDS_colorsR, ptcDS_colorsG, ptcDS_colorsB = cloud2array(pcd_ds)
# ---------------------------------------------------
#   Create messages to send
# ---------------------------------------------------
ptc_msg = cloudArray()
ptcDS_msg = cloudArray()
ptc_msg.pointsX = ptc_pointsX
ptc_msg.pointsY = ptc_pointsY
ptc_msg.pointsZ = ptc_pointsZ
ptc_msg.colorsR = ptc_colorsR
ptc_msg.colorsG = ptc_colorsG
ptc_msg.colorsB = ptc_colorsB

ptcDS_msg.pointsX = ptcDS_pointsX
ptcDS_msg.pointsY = ptcDS_pointsY
ptcDS_msg.pointsZ = ptcDS_pointsZ
ptcDS_msg.colorsR = ptcDS_colorsR
ptcDS_msg.colorsG = ptcDS_colorsG
ptcDS_msg.colorsB = ptcDS_colorsB

def talker():
    rospy.init_node('receiver', anonymous=False)
    rate = rospy.Rate(frame_rate) # 10hz
    while not rospy.is_shutdown():
        pub_ptc.publish(ptc_msg)
        pub_ptcDS.publish(ptcDS_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass