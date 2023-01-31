#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from pointCloud_Transform import cloud2array, array2cloud

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


# Load file
filename = '../../../rgbd-scenes-v2/pcd/01.pcd'
original_pcd = o3d.io.read_point_cloud(filename)

ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB = cloud2array(original_pcd)
pcd = array2cloud(ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB)

# Visualize

o3d.visualization.draw_geometries([pcd],
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])