#!/usr/bin/env python3

import open3d as o3d
import rospy
from pointCloud_Transform import array2cloud
from savi_tp2.msg import cloudArray

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.4968247576504106, 2.2836352945191325, 0.87840679827947743 ],
			"boundingbox_min" : [ -2.5744585151435198, -2.1581489860671899, -0.60582068710203252 ],
			"field_of_view" : 60.0,
			"front" : [ 0.64259021703866903, 0.52569095376874997, 0.55742877041995087 ],
			"lookat" : [ 0.35993510810021934, 0.20028892156868539, 0.25558948566773715 ],
			"up" : [ -0.41838167468135773, -0.36874521998147031, 0.8300504424621673 ],
			"zoom" : 0.14000000000000001
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

# ---------------------------------------------------
#   Topics to subscribe
# ---------------------------------------------------
topic_ptc = '/tp2_savi/clouds/ptcDS'
vis = o3d.visualization.Visualizer()
vis.create_window()

def callback(data):
    # print(data.pointsX)
    cloud = array2cloud(data.pointsX, data.pointsY, data.pointsZ, data.colorsR, data.colorsG, data.colorsB)
    o3d.visualization.draw_geometries([cloud],
                                        zoom=view['trajectory'][0]['zoom'],
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'])
    
def listener():
    rospy.init_node('table_recognizer')
    rospy.Subscriber(topic_ptc, cloudArray, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

listener()