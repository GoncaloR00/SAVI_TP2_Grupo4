import rospy
from sensor_msgs.msg import PointCloud2
import pcl_msgs.msg

def pcl_publisher():
    rospy.init_node('pcl_publisher', anonymous=True)
    pub = rospy.Publisher('pcl', PointCloud2, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    # Load point cloud from PCD file
    cloud = pcl.load('example.pcd')

    # Convert point cloud to ROS message
    ros_cloud = pcl.ros.point_cloud2.to_ros_msg(cloud)
    ros_cloud.header.frame_id = "world"
    while not rospy.is_shutdown():
        pub.publish(ros_cloud)
        rate.sleep()

if __name__ == '__main__':
    try:
        pcl_publisher()
    except rospy.ROSInterruptException:
        pass
