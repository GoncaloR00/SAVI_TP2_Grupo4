#!/usr/bin/python3

import os
import pathlib
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from typing import Tuple, Dict, List
import torchvision
from TinyVGG import TinyVGG

MODEL_PATH = pathlib.Path("./models")

MODEL_RESULTS_NAME = "SAVI_modelResultsTest.pth"
MODEL_RESULTS_SAVE_PATH = MODEL_PATH / MODEL_RESULTS_NAME

MODEL_DICT_NAME = "SAVI_modelDictTest.pth"
MODEL_DICT_SAVE_PATH = MODEL_PATH / MODEL_DICT_NAME

MODEL_COMP_NAME = "SAVI_modelCompTest.pth"
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

def main():
    class_names = find_classes(train_dir)[0]
    train_len = len(next(os.walk(train_dir))[1])
    model_1_results = torch.load(f=MODEL_RESULTS_SAVE_PATH)
    plot_loss_curves(model_1_results)
    model_1 = TinyVGG(
            input_shape=3,
            hidden_units=10,
            output_shape=train_len)

    model_1_dict = torch.load(f=MODEL_DICT_SAVE_PATH)
    model_1.load_state_dict(model_1_dict)
    model_1.eval()
    custom_image_path = pathlib.Path('../../../rgbd-dataset-train/') / "apple" / "apple_1_1_1_crop.png"
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
    custom_image = custom_image / 255.
    custom_image_transformed = transform_image(custom_image)


    # exit()
    with torch.inference_mode():
            custom_image_pred = model_1(custom_image_transformed)
    # exit()

    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    custom_image_pred_class = class_names[custom_image_pred_label.cpu()]

    print(custom_image_pred_class)
    # # Teste
    # teste = torch.load(f=MODEL_SAVE_PATH)
    # model_1.load_state_dict(teste)
    #     test_loss, test_acc = test_step(model=model_1,
    #         dataloader=test_dataloader_simple,
    #         loss_fn=loss_fn)
    # print(f"loss:{test_loss}/nAcc:{test_acc}")
    # exit()

main()