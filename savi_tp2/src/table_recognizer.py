#!/usr/bin/env python3
# TODO Tentar usar deep learning
import open3d as o3d
import rospy
from pointCloud_Transform import array2cloud
from savi_tp2.msg import cloudArray, coord_transform
import numpy as np
from matplotlib import cm
from more_itertools import locate

# ---------------------------------------------------
#   Topics to subscribe
# ---------------------------------------------------
topic_ptc = '/tp2_savi/clouds/ptcDS'
# ---------------------------------------------------
#   Topics to publish
# ---------------------------------------------------
topic_table = '/tp2_savi/coords/trans2table'
pub_trans2table= rospy.Publisher(topic_table, coord_transform, queue_size=10)


# def callback(data):
#     print('got info')
#     cloud = array2cloud(data.pointsX, data.pointsY, data.pointsZ, data.colorsR, data.colorsG, data.colorsB)
def callback(data):
    # print('start')
    # Parameters
    num_planes = 2
    distance_threshold = 0.1
    ransac_n = 4
    num_iterations =120
    detected_plane_idx = []
    detected_plane_d = []
    cloud = array2cloud(data.pointsX, data.pointsY, data.pointsZ, data.colorsR, data.colorsG, data.colorsB)
    while True:
        plane_model, inliers = cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
        # Plane Model
        [a, b, c, d] = plane_model
        # If there is a plane that have de negative y, will be necessary make one more measurement/segmentation 
        if b < 0:
            num_planes = 3

        # Inlier Cloud
        inlier_cloud = cloud.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        
        # Segmetation pcd update
        outlier_cloud = cloud.select_by_index(inliers, invert=True)
        cloud = outlier_cloud

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
    cloud = detected_plane_idx[d_max_idx]
    
    # -------- Table Plane Clustering ----------
    # Clustering 
    cluster_idx = np.array(cloud.cluster_dbscan(eps=0.08, min_points=50))
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
        object_points = cloud.select_by_index(object_point_idx)
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
                    table_cloud = object['points']
    try:            
        center = table_cloud.get_center() # visual center of the table
        tx, ty, tz = center[0], center[1], center[2]
        #TODO Encontrar ângulos pelos vetores ortogonais
        rx = -108
        ry = 0
        rz = -37
        #TODO Determinar extremidades da mesa
        xmin = -0.6
        xmax = 0.6
        ymin= -0.6
        ymax = 0.6
        zmin = -0.02
        zmax = 0.4


        msg = coord_transform()
        msg.tx = -tx
        msg.ty = -ty
        msg.tz = -tz
        msg.rx = rx
        msg.ry = ry
        msg.rz = rz
        msg.xmin = xmin
        msg.xmax = xmax
        msg.ymin = ymin
        msg.ymax = ymax
        msg.zmin = zmin
        msg.zmax = zmax
        pub_trans2table.publish(msg)
        print('here')
    except Exception as e:
        print(e)
def listener():
    rospy.init_node('table_recognizer')
    rospy.Subscriber(topic_ptc, cloudArray, callback)
    # spin() simply keeps python from exiting until this node is stopped

listener()
rospy.spin()