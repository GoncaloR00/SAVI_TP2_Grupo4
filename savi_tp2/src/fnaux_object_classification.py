from fnaux_coord_transform import undoTransform
from pointCloud_Transform import cloud2array
import cv2
import numpy as np

def projectPoints(cloud, intrinsic, tx, ty, tz, roll, pitch, yaw):
    new_cloud = undoTransform(cloud, tx, ty, tz, roll, pitch, yaw)

    image_points, _ = cv2.projectPoints(objectPoints=np.asarray(new_cloud.points), cameraMatrix=intrinsic, tvec=np.zeros((3,1)), rvec = np.identity(3), distCoeffs=np.zeros(0))
    px = image_points[:,0,0].astype('int32')
    py = image_points[:,0,1].astype('int32')
    # py = image_points[:,1]
    px_max = max(px)
    py_max = max(py)
    px_min = min(px)
    py_min = min(py)
    image = np.ones((px_max - px_min, py_max - py_min, 3))
    image = image * 255
    for k in range(px.size - 1):
        # color = np.asarray(new_cloud.colors)[k,:] * 255
        image[px[k] - px_min -1, py[k]- py_min - 1, :] = np.asarray(new_cloud.colors)[k,:] * 255
    
    # print(f"\ncores:{np.asarray(new_cloud.colors).shape}\npixels:{py.size}")
    image = image.astype(np.uint8)
    cv2.imshow('teste', image)
    cv2.waitKey(0)
    # print(np.asarray(new_cloud.colors))
    # ptc_pointsX, ptc_pointsY, ptc_pointsZ, ptc_colorsR, ptc_colorsG, ptc_colorsB = cloud2array(new_cloud)