<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
<h3 align="center">J.A.R.V.I.S. 3000 Finder</h3>

  <p align="center">
    Incorporates the utilization of a deep neural network to categorize objects obtained from 3D models or RGB-D cameras.
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- Introduction -->
## Introduction

The J.A.R.V.I.S. 3000 Finder is an advanced object classification system that uses a deep neural network to identify objects captured from 3D models or RGB-D cameras. This project was developed for an Advanced Industrial Vision Systems class as the second evaluation. The objective is to detect and extract objects from a point cloud and then pass it through a classifier to determine the object's identity and physical characteristics, such as volume and area. 

<!-- ### Built With
* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
<p align="right">(<a href="#readme-top">back to top</a>)</p> -->


<!-- GETTING STARTED -->
## Getting Started

For this projet is used [ROS Noetic](http://wiki.ros.org/ROS/Installation) and the object classification CNN is built based on [PyTorch](https://pytorch.org/). To do object detection is used [Open3D](http://www.open3d.org/).

### Prerequisites

Create a new folder in scr folder in the ROS workspace (usually is *~/catkin_ws/src* )

Store the datasets in the created folder. You can download the datasets from the project's website. [here](https://rgbd-dataset.cs.washington.edu/).

Inside the JARVIS folder, there should be a structure similar to: (verificar)
  - models
  - rgbd-dataset
  - rgbd-scenes-v2
    - pc
    - imgs

### Installation
To install the project, clone the repository inside the folder in the *src* folder of your *catkin_ws* (same location as the datasets) and run the following lines:
```
git clone https://github.com/GoncaloR00/SAVI_TP2_Grupo4
cd ~/catkin_ws
catkin_make
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Organize the data
From the main folder of this repository, go to the script folder:
```
cd ./savi_tp2_train/src
```
And then run the script:
```
./organize_data.py
```
### Train Model
From the main folder of this repository, go to the script folder:
```
cd ./savi_tp2_train/src
```
And to train the model, run the code:

```
./train.py -n <number of epochs> -lr <learning rate>
```
This script will output 3 files:
- The complete model
- The dictionary of the model containing all parameters
- The results of the model

The names will be given based on the number of epochs

### Evaluate model
From the main folder of this repository, go to the script folder:
```
cd ./savi_tp2_train/src
```
If you want to see the loss and accuracy curves:
```
./visualize.py -n <number of epochs>
```
If you want to see an inference example on random images from the dataset:
```
./visualize.py -n <number of epochs> -infer
```
The right names will be green and the wrong ones will be red
***
### Get the object positions and classification
First, start the roscore:
```
roscore
```
Then start the speech and plot_image nodes:

Speech (from main folder of the repository):
```
cd ./savi_tp2/src
```
```
./text_to_speech.py
```
Image plot (from main folder of the repository):
```
cd ./savi_tp2/src
```
```
./Image_plot.py
```
And finally start the main node (from main folder of the repository):
```
cd ./savi_tp2/src
```
```
./compute_cloud.py -cl <cloud filename>
```
***

## References
https://github.com/miguelriemoliveira/savi_22-23

https://poloclub.github.io/cnn-explainer/

http://www.open3d.org/

http://wiki.ros.org/

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

André Ferreira - andrerferreira@ua.pt

Gonçalo Ribeiro - gribeiro@ua.pt

João Cruz - 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Professor Miguel Oliveira - mriem@ua.pt

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
