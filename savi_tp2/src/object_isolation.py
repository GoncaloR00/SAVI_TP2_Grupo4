#!/usr/bin/env python3
import open3d as o3d
import rospy
from pointCloud_Transform import array2cloud
from savi_tp2.msg import cloudArray, coord_transform
import numpy as np
from matplotlib import cm
from more_itertools import locate
import message_filters
# ---------------------------------------------------
#   Topics to subscribe
# ---------------------------------------------------
topic_ptc = '/tp2_savi/clouds/ptcDS'
topic_table = '/tp2_savi/coords/trans2table'

subs_ptc = message_filters.Subscriber(topic_ptc, cloudArray)
subs_table = message_filters.Subscriber(topic_table, coord_transform)

# The first message waits for the message of the other topic
ts_message = message_filters.TimeSynchronizer([subs_ptc, subs_table],10)

def Callback():
    print('now')

def listener():
    rospy.init_node('table_recognizer')
    ts_message.registerCallback(Callback)
    # spin() simply keeps python from exiting until this node is stopped
    
listener()
rospy.spin()
# ---------------------------------------------------
#   Topics to publish
# ---------------------------------------------------

# pub_trans2table= rospy.Publisher(topic_table, coord_transform, queue_size=10)

