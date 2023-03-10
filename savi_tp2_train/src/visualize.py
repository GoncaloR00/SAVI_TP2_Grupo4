#!/usr/bin/python3

import os
import pathlib
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from typing import Tuple, Dict, List
import torchvision
from TinyVGG import TinyVGG
import argparse
import random
import numpy as np
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(
                    prog = 'visualize',
                    description = 'This program is useful to see the results of the training trains a deep learning model to do the classification task in the compute_cloud program',
                    epilog = '-cl <cloud name>')
parser.add_argument('-n',dest='epoch', type=int)
parser.add_argument('-infer',dest='inference', action="store_true")
args = parser.parse_args()

n_epoch = args.epoch
infer = args.inference

MODEL_PATH = pathlib.Path("./models")

MODEL_RESULTS_NAME = "SAVI_modelResults" + str(n_epoch) + ".pth"
MODEL_RESULTS_SAVE_PATH = MODEL_PATH / MODEL_RESULTS_NAME

MODEL_DICT_NAME = "SAVI_modelDict" + str(n_epoch) + ".pth"
MODEL_DICT_SAVE_PATH = MODEL_PATH / MODEL_DICT_NAME

MODEL_COMP_NAME = "SAVI_modelComp" + str(n_epoch) + ".pth"
MODEL_COMP_SAVE_PATH = MODEL_PATH / MODEL_COMP_NAME

train_dir = pathlib.Path('../../../rgbd-dataset-train/')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

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

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.show()

def transform_image(custom_image):
    # Create transform pipeline to resize image
    custom_image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    image = custom_image_transform(custom_image)
    return image.unsqueeze(dim=0)


def make_prediction(model, transformed_image):
    model.eval()
    with torch.inference_mode():
        custom_image_pred = model(transformed_image.to(device))
    return custom_image_pred


def plot_images(image_list, class_pred_list, success_list):
    # Defina o n??mero de subplots
    n_rows, n_cols = 3, 3
    # Crie uma figura e adicione subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 7)) # figsize=(width, height)

    counter = 00
    # Itera sobre cada subplot e adiciona uma imagem com t??tulo e legenda
    for i in range(n_rows):
        for j in range(n_cols):
            if success_list[counter]:
                color = 'green'
            else:
                color = 'red'
            axs[i, j].imshow(image_list[counter]) # Adiciona uma imagem aleat??ria
            axs[i, j].set_title(f"{class_pred_list[counter]}", pad = -10,color= color) # pad = -10 para n??o sobrescrever o t??tulo
            axs[i, j].axis("off") # Desativa os eixos
            #axs[i, j].set_xlabel("Eixo X")
            #axs[i, j].set_ylabel("Eixo Y")
            counter += 1 # Incrementa o contador

    # Mostre a figura
    fig.canvas.set_window_title("Objetos em cena")
    plt.show()

def main():
    if infer:
        class_names = find_classes(train_dir)[0]
        pred_list = []
        image_list = []
        success_list = []

        for i in range(9):
            real_class = random.choice(class_names)
            images_path = pathlib.Path('../../../rgbd-dataset-all/') / real_class
            filename = random.choice(os.listdir(str(images_path)))
            custom_image_path = images_path / filename
            custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
            custom_image = custom_image / 255.
            custom_image_transformed = transform_image(custom_image)
            train_len = len(next(os.walk(train_dir))[1])
            model_1 = TinyVGG(
                    input_shape=3,
                    hidden_units=10,
                    output_shape=train_len)
            model_1_dict = torch.load(f=MODEL_DICT_SAVE_PATH)
            model_1.load_state_dict(model_1_dict)
            model_1.eval()
            with torch.inference_mode():
                custom_image_pred = model_1(custom_image_transformed)
            custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
            custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
            custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
            if custom_image_pred_class == real_class:
                success = True
            else:
                success = False
            pred_list.append(custom_image_pred_class)
            success_list.append(success)
            image_list.append(mpimg.imread(str(custom_image_path)))
        plot_images(image_list, pred_list, success_list)
    else:
        model_1_results = torch.load(f=MODEL_RESULTS_SAVE_PATH)
        plot_loss_curves(model_1_results)

main()