from fnaux_coord_transform import undoTransform
import cv2
import numpy as np
import os
from typing import Tuple, Dict, List
import torch
import pathlib
import torchvision
from torchvision import transforms
from TinyVGG import TinyVGG
import random

def projectPoints(cloud, intrinsic, tx, ty, tz, roll, pitch, yaw):
    # new_cloud = undoTransform(cloud, tx, ty, tz, roll, pitch, yaw)
    # Afastar a camera da mesa
    new_cloud = undoTransform(cloud, 0, 0, 0, 0, 0, -90)
    new_cloud = undoTransform(cloud, 0, 0, -1.5, 0, 120, 0)
    image_points, _ = cv2.projectPoints(objectPoints=np.asarray(new_cloud.points), cameraMatrix=intrinsic, tvec=np.zeros((3,1)), rvec = np.identity(3), distCoeffs=np.zeros(0))
    px = image_points[:,0,0].astype('int32')
    py = image_points[:,0,1].astype('int32')
    px_max = max(px)
    py_max = max(py)
    px_min = min(px)
    py_min = min(py)
    # print(f"px_min = {px_min}, px_max = {px_max}, py_min = {py_min}, py_max = {py_max},")
    # image = np.ones((py_max - py_min, px_max - px_min, 3))
    image = np.ones((px_max - px_min, py_max - py_min, 3))
    image = image * 255
    for k in range(px.size - 1):
        # image[py[k]- py_min - 1, px[k] - px_min -1, :] = np.asarray(new_cloud.colors)[k,:] * 255
        image[px[k] - px_min -1,py[k]- py_min - 1 , :] = np.asarray(new_cloud.colors)[k,:] * 255
    # print(f"\ncores:{np.asarray(new_cloud.colors).shape}\npixels:{py.size}")
    image = image.astype(np.uint8)
    return image

def projectPointsMODE(cloud, intrinsic, tx, ty, tz, roll, pitch, yaw, original_image):
    # Colocar a camera na posição original
    new_cloud = undoTransform(cloud, tx, ty, tz, roll, pitch, yaw)
    image_points, _ = cv2.projectPoints(objectPoints=np.asarray(new_cloud.points), cameraMatrix=intrinsic, tvec=np.zeros((3,1)), rvec = np.identity(3), distCoeffs=np.zeros(0))
    px = image_points[:,0,0].astype('int32')
    py = image_points[:,0,1].astype('int32')
    px_max = max(px)
    py_max = max(py)
    px_min = min(px)
    py_min = min(py)
    image = original_image[py_min:py_max, px_min:px_max]
    # Teste ao classificador
    # real_class = 'banana'
    # images_path = pathlib.Path('../../../rgbd-dataset-all/') / real_class
    # filename = random.choice(os.listdir(str(images_path)))
    # custom_image_path = images_path / filename
    # image = cv2.imread(str(custom_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def transform_image(custom_image):
    # Create transform pipeline to resize image
    custom_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ])
    image = custom_image_transform(custom_image)
    return image.unsqueeze(dim=0)

def classifyImage(image,train_dir,MODEL_DICT_SAVE_PATH):
    class_names = find_classes(train_dir)[0]
    train_len = len(next(os.walk(train_dir))[1])
    image = image.astype(np.float32)
    image = image / 255.
    model_1 = TinyVGG(
            input_shape=3,
            hidden_units=10,
            output_shape=train_len)
    model_1_dict = torch.load(f=MODEL_DICT_SAVE_PATH)
    model_1.load_state_dict(model_1_dict)
    model_1.eval()
    custom_image_transformed = transform_image(image)
    with torch.inference_mode():
            custom_image_pred = model_1(custom_image_transformed)
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
    return custom_image_pred_class

def object_classification(cloud_list, intrinsic, tx, ty, tz, roll, pitch, yaw, train_dir, MODEL_DICT_SAVE_PATH):
    image_list = []
    classification_list = []
    for cloud in cloud_list:
        image_list.append(projectPoints(cloud, intrinsic, tx, ty, tz, roll, pitch, yaw))
    for image in image_list:
        classification_list.append(classifyImage(image,train_dir,MODEL_DICT_SAVE_PATH))
    return image_list, classification_list

def object_classificationMODE(cloud_list, intrinsic, tx, ty, tz, roll, pitch, yaw, train_dir, MODEL_DICT_SAVE_PATH, original_image):
    image_list = []
    classification_list = []
    for cloud in cloud_list:
        image_list.append(projectPointsMODE(cloud, intrinsic, tx, ty, tz, roll, pitch, yaw, original_image))
    for image in image_list:
        classification_list.append(classifyImage(image,train_dir,MODEL_DICT_SAVE_PATH))
    return image_list, classification_list