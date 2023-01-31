#!/usr/bin/env python3
import pathlib
import open3d as o3d
import rospy
from fnaux_table_recognizer import table_recognizer
from fnaux_coord_transform import transform, undoTransform
from pointCloud_Transform import cloud2array
from fnaux_object_isolation import object_isolation
from fnaux_object_classification import object_classification, object_classificationMODE
import numpy as np
import cv2

#   SELECT CLOUD
cloud_name = '06.pcd'
cloud_name_number = cloud_name.split('.')[0]
cloud_number = int(cloud_name_number)
if (cloud_number >= 5 and cloud_number <= 8) or cloud_number == 13:
    mode = False
else:
    mode = True
# ---------------------------------------------------
#   Parameters for intrinsic camera matrix
# ---------------------------------------------------
fx = 570.3
fy = 570.3
cx = 320
cy = 240
train_dir = pathlib.Path('../../../rgbd-dataset-train/')
MODEL_PATH = pathlib.Path("./models")
MODEL_DICT_NAME = "SAVI_modelDict500.pth"
MODEL_DICT_SAVE_PATH = MODEL_PATH / MODEL_DICT_NAME
# ---------------------------------------------------
#   Load file (for starters is only for testing - might be included in the loop to change over time)
# ---------------------------------------------------
path = pathlib.Path('../../../rgbd-scenes-v2/pcd')
filename = str(path / cloud_name)
original_pcd = o3d.io.read_point_cloud(filename)
original_image = cv2.imread(str(path / (cloud_name_number + '.png')))
# old_pcd = copy.deepcopy(original_pcd)
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

entities = []
object_cloud_list = []
object_array_list = []
for object in isolated_objects:
    bbox = object['bbox']
    entities.append(bbox)
    object_cloud = original_pcd.crop(bbox)
    ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB = cloud2array(object_cloud)
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
for idx, classification in enumerate(classification_list):
    cv2.imshow(classification + str(idx), cv2.cvtColor(image_list[idx],cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
print(classification_list)
# teste = projectPoints(object_cloud_list[1], intrinsic, tx, ty, tz, roll, pitch, yaw)


# exit()
# Visualization for testing


    # object_array_list.append(object_array)
frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0, 0, 0]))
entities.append(frame)
# new_pcl = copy.deepcopy(original_pcd)
# new_pcl = undoTransform(new_pcl, 0, 0, -1, 0, 120, 0)
# pcl = np.concatenate((entities, [new_pcl]))
pcl = np.concatenate((entities, [original_pcd]))
# pcl = [object_cloud_list[0]]
# print(object_array_list[0])
# exit()
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
			"lookat" : [ 0, 0, 0 ],
			"up" : [ -0.41838167468135773, -0.36874521998147031, 0.8300504424621673 ],
			"zoom" : 0.14000000000000001
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

o3d.visualization.draw_geometries(pcl,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])