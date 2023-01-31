#!/usr/bin/env python3
import rospy
from savi_tp2.msg import allData

# ---------------------------------------------------
#   RosTopic to subscribe
# ---------------------------------------------------
topic_ptc = '/tp2_savi/ptc'
pub_allData = rospy.Publisher(topic_ptc, allData, queue_size=10)

def callback(message):
    print(message.images)

def listener():
    rospy.init_node('visualizer')
    rospy.Subscriber(topic_ptc, allData, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == "__main__":
    listener()