import open3d as o3d
import numpy as np

def cloud2array(cloud):
    ptc_points = np.asarray(cloud.points)
    ptc_colors = np.asarray(cloud.colors)
    ptc_pointsX = ptc_points[:,0]
    ptc_pointsY = ptc_points[:,1]
    ptc_pointsZ = ptc_points[:,2]
    ptc_colorsR = ptc_colors[:,0]
    ptc_colorsG = ptc_colors[:,1]
    ptc_colorsB = ptc_colors[:,2]
    return ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB
    
def array2cloud(ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB):
    size = ptc_pointsX.size
    points = np.zeros((size,3))
    colors = np.zeros((size,3))
    points[:,0] = ptc_pointsX
    points[:,1] = ptc_pointsY
    points[:,2] = ptc_pointsZ
    colors[:,0] = ptc_colorsR
    colors[:,1] = ptc_colorsG
    colors[:,2] = ptc_colorsB
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud