import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate

def crop(cloud, x_min, y_min, z_min, x_max, y_max, z_max):

    # Table region
    np_points = np.ndarray((8,3), dtype=float)
    np_points[0, :] = [x_min, y_min, z_min]
    np_points[1, :] = [x_max, y_min, z_min]
    np_points[2, :] = [x_max, y_max, z_min]
    np_points[3, :] = [x_min, y_max, z_min]

    np_points[4, :] = [x_min, y_min, z_max]
    np_points[5, :] = [x_max, y_min, z_max]
    np_points[6, :] = [x_max, y_max, z_max]
    np_points[7, :] = [x_min, y_max, z_max]

    bbox_points = o3d.utility.Vector3dVector(np_points)
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points)
        # bbox.color = (0, 1, 0)
    table_cloud = cloud.crop(bbox)
    return table_cloud

def findTableTop(cloud_table):
    distance_threshold=0.01
    ransac_n=3
    num_iterations=100
    plane_model, inliers = cloud_table.segment_plane(distance_threshold,ransac_n, num_iterations)
    [a, b, c, d] = plane_model
    # inliers = cloud_table.select_by_index(inliers)
    outlier_cloud = cloud_table.select_by_index(inliers, invert=True)
    return outlier_cloud

def cluster_objects(objects_cloud):
    # Clustering 
    cluster_idx = np.array(objects_cloud.cluster_dbscan(eps=0.030, min_points=60, print_progress=True))
        
    # Clusters Index
    objects_idx = list(set(cluster_idx))

    # Remove noise 
    objects_idx.remove(-1) 

    colormap = cm.Pastel1(list(range(0,len(objects_idx))))
    objects=[]
    for object_idx in objects_idx:
        object_point_idx = list(locate(cluster_idx, lambda X: X== object_idx))
        object_points = objects_cloud.select_by_index(object_point_idx)
        object_center = object_points.get_center()
        # Create a dictionary to represent objects
        d = {}
        d['idx'] = str(objects_idx[object_idx])
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        #d['points'].paint_uniform_color(d['color'])
        d['center'] = object_center
        #print('center of object' + str(object_idx) + ': ' + str(object_center))


        # -------------- STDeviation of each coordinate about XY and Z Axis ------------------

        
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
        obj_bbox.color = (0,1,0)
        d['bbox'] = obj_bbox
        objects.append(d)
    for object in objects:      
        object['points'].paint_uniform_color(object['color'])
    return objects
        
def object_isolation(cloud, x_min, y_min, z_min, x_max, y_max, z_max):
    cloud_table = crop(cloud, x_min, y_min, z_min, x_max, y_max, z_max)
    objects_cloud = findTableTop(cloud_table)
    objects = cluster_objects(objects_cloud)
    return objects

