#!/usr/bin/env python3
import pathlib
import open3d as o3d
import open3d.visualization.gui as gui
import rospy
from savi_tp2.msg import cloudArray, images, bboxes
from fnaux_table_recognizer import table_recognizer
from fnaux_coord_transform import transform, undoTransform
from pointCloud_Transform import cloud2array, cloud2BIGarray
from fnaux_object_isolation import object_isolation
from fnaux_object_classification import object_classification, object_classificationMODE
import open3d.visualization.rendering as rendering
import numpy as np
import cv2
from cv_bridge import CvBridge
import argparse

parser = argparse.ArgumentParser(
                    prog = 'compute_cloud',
                    description = 'This program detect and classifies the object on the table from a rgbd point cloud',
                    epilog = '-cl <cloud name>')
parser.add_argument('-cl',dest='name', type=str)
args = parser.parse_args()
cloud_name = args.name

def main(cloud_name):
	#   SELECT CLOUD
	# cloud_name = '01.pcd'
	cloud_name_number = cloud_name.split('.')[0]
	cloud_number = int(cloud_name_number)
	if (cloud_number >= 5 and cloud_number <= 8) or cloud_number == 13:
		mode = False
	else:
		mode = True
	# ---------------------------------------------------
	#   Load file (for starters is only for testing - might be included in the loop to change over time)
	# ---------------------------------------------------
	path = pathlib.Path('../../../rgbd-scenes-v2/pcd')
	filename = str(path / cloud_name)
	original_pcd = o3d.io.read_point_cloud(filename)
	original_image = cv2.imread(str(path / (cloud_name_number + '.png')))
	# ---------------------------------------------------
	#   RosTopic to publish
	# ---------------------------------------------------
	topic_images = '/tp2_savi/images'
	pub_image = rospy.Publisher(topic_images, images, queue_size=5)
	# Classification
	train_dir = pathlib.Path('../../../rgbd-dataset-train/')
	MODEL_PATH = pathlib.Path("./models")
	MODEL_DICT_NAME = "SAVI_modelDict500.pth"
	MODEL_DICT_SAVE_PATH = MODEL_PATH / MODEL_DICT_NAME
	# ---------------------------------------------------
	#   Parameters for intrinsic camera matrix
	# ---------------------------------------------------
	fx = 570.3
	fy = 570.3
	cx = 320
	cy = 240
	# ---------------------------------------------------
	#   Create messages
	# ---------------------------------------------------
	ptc_msg = cloudArray()
	# ---------------------------------------------------
	#   Downscale cloud
	# ---------------------------------------------------
	pcd_ds = original_pcd.voxel_down_sample(voxel_size=0.01)
	# ---------------------------------------------------
	#   Get table center (original frame), table limits and frame rotations
	# ---------------------------------------------------
	tx, ty, tz, roll, pitch, yaw, xmin, xmax, ymin, ymax, zmin, zmax = table_recognizer(pcd_ds)
	# ---------------------------------------------------
	#   Apply new coordinate system to the point clouds
	# ---------------------------------------------------
	original_pcd = transform(original_pcd, tx, ty, tz, roll, pitch, yaw)
	pcd_ds = transform(pcd_ds, tx, ty, tz, roll, pitch, yaw)
	# ---------------------------------------------------
	#   Get objects positions and bounding boxes
	# ---------------------------------------------------
	isolated_objects = object_isolation(pcd_ds, xmin, xmax, ymin, ymax, zmin, zmax)

	object_cloud_list = []
	entities = []
	object_array_list = []
	# dim = len(isolated_objects)
	# for object in isolated_objects:
	# 	x_min, y_min, z_min, x_max, y_max, z_max = object['boundaries']
	# 	bbox_xmin.append(x_min)
	# 	bbox_ymin.append(y_min)
	# 	bbox_zmin.append(z_min)
	# 	bbox_xmax.append(x_max)
	# 	bbox_ymax.append(y_max)
	# 	bbox_zmax.append(z_max)
	# print(bbox_xmin)
	for object in isolated_objects:
		bbox = object['bbox']
		entities.append(bbox)
		object_cloud = original_pcd.crop(bbox)
		# ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB = cloud2array(object_cloud)
		object_cloud_list.append(object_cloud)
	# ---------------------------------------------------
	#   Objects classification
	# ---------------------------------------------------
	intrinsic = np.zeros((3,3))
	intrinsic[0,0] = fx
	intrinsic[1,1] = fy
	intrinsic[0,2] = cx
	intrinsic[1,2] = cy
	intrinsic[2,2] = 1

	# imagem = projectPoints(original_pcd, intrinsic, tx, ty, tz, roll, pitch, yaw)
	# cv2.imshow('Nuvem', imagem)
	# cv2.waitKey(0)
	if mode:
		image_list, classification_list = object_classificationMODE(object_cloud_list, intrinsic, tx, ty, tz, roll, pitch, yaw, train_dir, MODEL_DICT_SAVE_PATH,original_image)
	else:
		image_list, classification_list = object_classification(object_cloud_list, intrinsic, tx, ty, tz, roll, pitch, yaw, train_dir, MODEL_DICT_SAVE_PATH)
	print(classification_list)
	# Images
	bridge = CvBridge()
	transformed_images = []
	for image in image_list:
		transformed_images.append(bridge.cv2_to_imgmsg(image, "passthrough"))
	# message.images = transformed_images

	# Point Cloud
	# ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB = cloud2array(original_pcd)
	# pcl = np.concatenate((bboxes, [original_pcd]))
	# print(bboxes)
	# ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB = cloud2array(bboxes)
	# ptc_msg.pointsX = ptc_pointsX
	# ptc_msg.pointsY = ptc_pointsY
	# ptc_msg.pointsZ = ptc_pointsZ
	# ptc_msg.colorsR = ptc_colorsR
	# ptc_msg.colorsG = ptc_colorsG
	# ptc_msg.colorsB = ptc_colorsB
	# # ptc_msg.BBox_xmin = bbox_xmin
	# # ptc_msg.BBox_ymin = bbox_ymin
	# # ptc_msg.BBox_zmin = bbox_zmin
	# # ptc_msg.BBox_xmax = bbox_xmax
	# # ptc_msg.BBox_ymax = bbox_ymax
	# # ptc_msg.BBox_zmax = bbox_zmax

	# pub_ptc.publish(ptc_msg)
	# print('sended')
	# exit()
	# for idx, classification in enumerate(classification_list):
	# 	cv2.imshow(classification + str(idx), cv2.cvtColor(image_list[idx],cv2.COLOR_RGB2BGR))
	# cv2.waitKey(0)
	# print(classification_list)
	# # teste = projectPoints(object_cloud_list[1], intrinsic, tx, ty, tz, roll, pitch, yaw)


	# # exit()
	# Visualization for testing


		# object_array_list.append(object_array)
	# frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0, 0, 0]))
	# entities.append(frame)
	# # new_pcl = copy.deepcopy(original_pcd)
	# # new_pcl = undoTransform(new_pcl, 0, 0, -1, 0, 120, 0)
	# pcl = np.concatenate((entities, [new_pcl]))
	pcl = np.concatenate((entities, [original_pcd]))
	# pcl = [object_cloud_list[0]]
	# print(object_array_list[0])
	# exit()
	# view = {
	# 	"class_name" : "ViewTrajectory",
	# 	"interval" : 29,
	# 	"is_loop" : False,
	# 	"trajectory" : 
	# 	[
	# 		{
	# 			"boundingbox_max" : [ 2.4968247576504106, 2.2836352945191325, 0.87840679827947743 ],
	# 			"boundingbox_min" : [ -2.5744585151435198, -2.1581489860671899, -0.60582068710203252 ],
	# 			"field_of_view" : 60.0,
	# 			"front" : [ 0.64259021703866903, 0.52569095376874997, 0.55742877041995087 ],
	# 			"lookat" : [ 0, 0, 0 ],
	# 			"up" : [ -0.41838167468135773, -0.36874521998147031, 0.8300504424621673 ],
	# 			"zoom" : 0.14000000000000001
	# 		}
	# 	],
	# 	"version_major" : 1,
	# 	"version_minor" : 0
	# }

	# o3d.visualization.draw_geometries(bboxes,
	# 									zoom=view['trajectory'][0]['zoom'],
	# 									front=view['trajectory'][0]['front'],
	# 									lookat=view['trajectory'][0]['lookat'],
	# 									up=view['trajectory'][0]['up'])
	
	app = gui.Application.instance
	app.initialize() # create a open3d app
	w = app.create_window("Detected Objects", 1980, 1080)
	material = rendering.Material() #rendering.materialrecord (outras versoes de open3d)
	material.shader = "defaultUnlit"
	material.point_size = 3 * w.scaling
	widget3d = gui.SceneWidget()
	widget3d.scene = rendering.Open3DScene(w.renderer)
	widget3d.scene.set_background([1,1,1,1])
	for entity_idx, entity in enumerate(pcl):
		widget3d.scene.add_geometry("Entity" + str(entity_idx),entity, material)
		for obj_idx, obj in enumerate(isolated_objects):
			l = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.11)), 'Object ' + str(obj['idx']) + ':'+ str(classification_list[obj_idx]))
			#volume em (x x y x z) mm
			#l2 = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.06)), 'Volume: ( ' + str(round(obj['x_width']*1000,0)) + 
			#                           ' x ' + str(round(obj['y_width']*1000,0)) + ' x ' + str(round(obj['height']*1000,0)) + ') mm' )
			#area em mm2
			l3 = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.08)), 'Area: (' + str(round(obj['area']* 10000, 0)) + ') cm2')
			#volume em mm3
			l2 = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.05)), 'Volume: (' + str(round(obj['volume']*1000000,0)) + ') cm3')
			#cor label
			l.color = gui.Color(obj['color'][0], obj['color'][1], obj['color'][2],)
			l3.color = gui.Color(0, 1, 1,)
			l2.color = gui.Color(0, 1, 1,)

	bbox = widget3d.scene.bounding_box
	widget3d.setup_camera(60.0, bbox, bbox.get_center())
	w.add_child(widget3d)
	app.run()
if __name__ == "__main__":
	# rospy.init_node('compute_cloud')
	main(cloud_name)