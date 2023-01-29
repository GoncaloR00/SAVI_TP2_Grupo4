#!/usr/bin/python3

import open3d
import numpy as np
from ctypes import * # convert float to uint32

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

dir = './pcd_point_cloud.pcd'


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
        cloud_data=np.c_[points, colors]
    
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

    # -- Example of usage
if __name__ == "__main__":
    rospy.init_node('test_pc_conversion_between_Open3D_and_ROS', anonymous=True)
    # -- Read point cloud from file
    open3d_cloud = open3d.io.read_point_cloud(dir)
    rospy.loginfo("Loading cloud from file by open3d.read_point_cloud: ")
    print(open3d_cloud)
    print("")

    # -- Set publisher
    topic_name="kinect2/qhd/points"
    pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)

     # -- Convert open3d_cloud to ros_cloud, and publish. Until the subscribe receives it.
    while True:
        if 1: # Use the cloud from file
            rospy.loginfo("Converting cloud from Open3d to ROS PointCloud2 ...")
            ros_cloud = convertCloudFromOpen3dToRos(open3d_cloud)

        else: # Use the cloud with 3 points generated below
            rospy.loginfo("Converting a 3-point cloud into ROS PointCloud2 ...")
            TEST_CLOUD_POINTS = [
                [1.0, 0.0, 0.0, 0xff0000],
                [0.0, 1.0, 0.0, 0x00ff00],
                [0.0, 0.0, 1.0, 0x0000ff],
            ]
            ros_cloud = pc2.create_cloud(
                Header(frame_id="odom"), FIELDS_XYZ , TEST_CLOUD_POINTS)

        # publish cloud
        pub.publish(ros_cloud)
        rospy.loginfo("Conversion and publish success ...\n")
        rospy.sleep(1)