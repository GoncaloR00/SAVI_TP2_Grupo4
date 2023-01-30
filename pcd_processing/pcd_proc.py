#!/usr/bin/env python3


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from more_itertools import locate
import pandas as pd
from matplotlib import cm
import math
import copy
import os



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


class PointCloudProcessing():

    def __init__ (self):
        pass

    def loadPointCloud (self, filename):

        print('Loading Point Cloud from ' + filename)
        self.pcd = o3d.io.read_point_cloud(filename)
        self.original_pcd = copy.deepcopy(self.pcd) #backup original pcd
        

    def downsample(self):

        # Pre Processing with Voxel downsampling
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.01)
        print('Downsampling reduced Point Cloud from ' + str(len(self.original_pcd.points)) + ' to ' + str(len(self.pcd.points)) + ' points')

    def frameadjustment(self, distance_threshold=0.1, ransac_n=4, num_iterations=120):
        
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1, origin=np.array([0, 0, 0]))

        # -------- Segmentation ----------
        
        # Segmentation Vars
        table_pcd = self.pcd
        num_planes = 2
        detected_plane_idx = []
        detected_plane_d = []
        

        while True:
            # Plane Segmentation
            plane_model, inliers = table_pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

            # Plane Model
            [a, b, c, d] = plane_model
            
            # If there is a plane that have de negative y, will be necessary make one more measurement/segmentation 
            if b < 0:
                num_planes = 3

            # Inlier Cloud
            inlier_cloud = table_pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            
            # Segmetation pcd update
            outlier_cloud = table_pcd.select_by_index(inliers, invert=True)
            table_pcd = outlier_cloud

            # Append detected plane
            if b > 0:
                detected_plane_idx.append(inlier_cloud)
                detected_plane_d.append(d)

            # Condition to stop pcd segmetation (2 measurments/segmentations)
            if len(detected_plane_idx) >= num_planes: 
                num_planes = 2
                break
        
        # Find idx of the table plane 
        d_max_idx = min(range(len(detected_plane_d)), key=lambda i: abs(detected_plane_d[i]-0))
        table_pcd = detected_plane_idx[d_max_idx]
        
        # -------- Table Plane Clustering ----------
        # Clustering 
        cluster_idx = np.array(table_pcd.cluster_dbscan(eps=0.08, min_points=50))
        objects_idx = list(set(cluster_idx))


        if cluster_idx.any() == -1:
            objects_idx.remove(-1)  
        
        # -------- Planes Caracterization ----------

        # Colormap
        colormap = cm.Set2(list(range(0,len(objects_idx))))

        # Caracterize all planes found to proceed to table detection/isolation 
        objects=[]
        for object_idx in objects_idx:
            
            object_point_idx = list(locate(cluster_idx, lambda X: X== object_idx))
            object_points = table_pcd.select_by_index(object_point_idx)
            object_center = object_points.get_center()

            # Create a dictionary to represent all planes
            d = {}
            d['idx'] = str(objects_idx)
            d['points'] = object_points
            d['color'] = colormap[object_idx, 0:3]
            d['points'].paint_uniform_color(d['color'])
            d['center'] = object_center
            
            objects.append(d)

        # -------- Table Selection ----------

        # The table is deteted with the comparison between the coordinates of centers (pcd and frame) and need to have more than 10000 points
        tables_to_draw=[]
        minimum_mean_xy = 1000
        
        for object in objects:
            tables_to_draw.append(object['points'])
            mean_x = object['center'][0]
            mean_y = object['center'][1]
            mean_z = object['center'][2]
            mean_xy = abs(mean_x) + abs(mean_y)
            if mean_xy < minimum_mean_xy:
                minimum_mean_xy = mean_xy
                if len(np.asarray(object['points'].points)) > 12000:
                    
                    self.table_cloud = object['points']
                    
        
        center = self.table_cloud.get_center() # visual center of the table
        tx, ty, tz = center[0], center[1], center[2] 

        return(-tx, -ty, -tz)


    def transform(self, r, p, y, tx, ty, tz):
    
        # Rad to Deg
        r = math.pi * r/180.0
        p = math.pi * p/180.0
        y = math.pi * y/180.0

        # Rotation
        rotation = self.pcd.get_rotation_matrix_from_xyz((r, p, y))
        self.pcd.rotate(rotation, center=(0, 0, 0))
        # Translate
        self.pcd = self.pcd.translate((tx, ty, tz))

        
    def croppcd(self, min_x, min_y, min_z, max_x, max_y, max_z):
        
        # Bounding box 
        np_points = np.ndarray((8,3), dtype=float)
        

        np_points[0, :] = [min_x, min_y, min_z]
        np_points[1, :] = [max_x, min_y, min_z]
        np_points[2, :] = [max_x, max_y, min_z]
        np_points[3, :] = [min_x, max_y, min_z]

        np_points[4, :] = [min_x, min_y, max_z]
        np_points[5, :] = [max_x, min_y, max_z]
        np_points[6, :] = [max_x, max_y, max_z]
        np_points[7, :] = [min_x, max_y, max_z]

        #Create AABB from points
        
        bbox_points = o3d.utility.Vector3dVector(np_points)

        self.bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points)
        self.bbox.color = (1, 0, 1)
        self.pcd = self.pcd.crop(self.bbox)
       

        #plane of the table
    def planesegmentation(self, distance_threshold=0.01, ransac_n=5, num_iterations=120):
        
        plane_model, inliers = self.pcd.segment_plane(distance_threshold,ransac_n, num_iterations)        
        [a, b, c, d] = plane_model
        
        self.inliers = self.pcd.select_by_index(inliers)
        self.outlier_cloud = self.pcd.select_by_index(inliers, invert=True)
        
        # Clustering 
    def pcd_clustering(self):
        
        cluster_idx = np.array(self.outlier_cloud.cluster_dbscan(eps=0.030, min_points=60, print_progress=True))
        
        objects_idx = list(set(cluster_idx)) # Clusters Index
        objects_idx.remove(-1)  

        number_of_objects = len(objects_idx)
        colormap = cm.Set2(list(range(0,number_of_objects)))


        objects=[]
        for object_idx in objects_idx:
            
            object_point_idx = list(locate(cluster_idx, lambda X: X== object_idx))
            object_points = self.outlier_cloud.select_by_index(object_point_idx)
            object_center = object_points.get_center()
            # Create a dictionary to represent objects
            d = {}
            d['idx'] = str(objects_idx[object_idx])
            d['points'] = object_points
            d['color'] = colormap[object_idx, 0:3]
            d['center'] = object_center

            print('center of object' + str(object_idx) + ': ' + str(object_center))
            
            


            # -------------- BBox of objets ------------------


            points = len(np.asarray(object_points.points))
            x_coordinates = []
            y_coordinates = []
            z_coordinates = []
        
            
            for num in range(points):
                x_coordinates.append(np.asarray(object_points.points[num][0]))
                y_coordinates.append(np.asarray(object_points.points[num][1]))
                z_coordinates.append(np.asarray(object_points.points[num][2]))

            x_width = max(x_coordinates) - min(x_coordinates)
            y_width = max(y_coordinates) - min(y_coordinates)
            z_height = max(z_coordinates) - min(z_coordinates)
            
            d['x_width'] =  x_width
            d['y_width'] = y_width
            d['height'] = z_height

            #area and volume bbox objetos
            d['area'] = x_width * y_width
            d['volume'] = x_width * y_width * z_height
                
            # BBOX
            np_points = np.ndarray((8,3), dtype=float)
            np_points[0, :] = [min(x_coordinates), min(y_coordinates), min(z_coordinates)]
            np_points[1, :] = [max(x_coordinates), min(y_coordinates), min(z_coordinates)]
            np_points[2, :] = [max(x_coordinates), max(y_coordinates), min(z_coordinates)]
            np_points[3, :] = [min(x_coordinates), max(y_coordinates), min(z_coordinates)]

            np_points[4, :] = [min(x_coordinates), min(y_coordinates), max(z_coordinates)]
            np_points[5, :] = [max(x_coordinates), min(y_coordinates), max(z_coordinates)]
            np_points[6, :] = [max(x_coordinates), max(y_coordinates), max(z_coordinates)]
            np_points[7, :] = [min(x_coordinates), max(y_coordinates), max(z_coordinates)]

        
            bbox_points = o3d.utility.Vector3dVector(np_points)

            obj_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points)
            obj_bbox.color = (0,0,0)
            d['bbox'] = obj_bbox
                      
            objects.append(d)
            

        # Pass value to attributes of the class
        self.objects_properties = objects
    
        self.objects_to_draw=[]

        # to draw each object already separated
        for object in objects:
            
            object['points'].paint_uniform_color(object['color'])
            self.objects_to_draw.append(object['points'])
            self.objects_to_draw.append(object['bbox'])

