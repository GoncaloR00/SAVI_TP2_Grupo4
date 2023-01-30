import math
def transform(cloud, tx, ty, tz, roll, pitch, yaw):
    # Rad to Deg
    r = math.pi * roll/180.0
    p = math.pi * pitch/180.0
    y = math.pi * yaw/180.0
    # Translate
    cloud = cloud.translate((tx, ty, tz))
    # Rotation
    rotation_r = cloud.get_rotation_matrix_from_xyz((r, 0, 0))
    cloud = cloud.rotate(rotation_r, center=(0, 0, 0))
    # Rotation
    rotation_p = cloud.get_rotation_matrix_from_xyz((0, p, 0))
    cloud = cloud.rotate(rotation_p, center=(0, 0, 0))
    # Rotation
    rotation_y = cloud.get_rotation_matrix_from_xyz((0, 0, y))
    cloud = cloud.rotate(rotation_y, center=(0, 0, 0))
    return cloud

def undoTransform(cloud, tx, ty, tz, roll, pitch, yaw):
    # Rad to Deg
    r = math.pi * roll/180.0
    p = math.pi * pitch/180.0
    y = math.pi * yaw/180.0
    # Rotation
    rotation_y = cloud.get_rotation_matrix_from_xyz((0, 0, -y))
    cloud = cloud.rotate(rotation_y, center=(0, 0, 0))
    # Rotation
    rotation_p = cloud.get_rotation_matrix_from_xyz((0, -p, 0))
    cloud = cloud.rotate(rotation_p, center=(0, 0, 0))
    # Rotation
    rotation_r = cloud.get_rotation_matrix_from_xyz((-r, 0, 0))
    cloud = cloud.rotate(rotation_r, center=(0, 0, 0))
    # Translate
    cloud = cloud.translate((-tx, -ty, -tz))
    return cloud