#!/usr/bin/python3
import rospy
import open3d_conversions
import open3d as o3d
from sensor_msgs.msg import PointCloud2

rospy.init_node('open3d_conversions_example')

current_cloud = o3d.io.read_point_cloud('pcd_point_cloud.pcd')

def handle_pointcloud(pointcloud2_msg):
    global current_cloud
    current_cloud = pointcloud2_msg

rate = rospy.Rate(1)

listener = rospy.Subscriber('/some_rgbd_camera/depth_registered/points', PointCloud2, handle_pointcloud, queue_size=1)
publisher = rospy.Publisher('~processed_point_cloud', PointCloud2, queue_size=1)

while not rospy.is_shutdown():
    if current_cloud is None:
        continue

    o3d_cloud = open3d_conversions.from_msg(current_cloud)

    # do open3d things
    # ...

    ros_cloud = open3d_conversions.to_msg(o3d_cloud, frame_id=current_cloud.header.frame_id, stamp=current_cloud.header.stamp)
        
    current_cloud = None
    rate.sleep()