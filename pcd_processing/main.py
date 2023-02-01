#!/usr/bin/env python3

import threading
import os
import cv2
import copy
import pyttsx3
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from gtts import gTTS
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
from pcd_proc import PointCloudProcessing
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering



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

def main():


    # ------------------------------------
    # Initialization 
    # ------------------------------------

    print('Starting Scene 3D Processing...\n')
    
    # Load PCD
    p = PointCloudProcessing()
    p.loadPointCloud('/home/andre/catkin_ws/src/SAVI_TP2_Grupo4/pcd_processing/pcds/07.ply')   
    

    
    # ------------------------------------
    # Execution 
    # ------------------------------------

    # Pre Processing with Voxel downsampling to increase process velocity
    p.downsample()

    # Calculation of the reference transformation parameters for the center of the table - In this case only for TRANS
    tx, ty, tz = p.frameadjustment()        
  
    # Transform 
    p.transform(0, 0, 0, tx, ty, tz)
    p.transform(-108, 0, 0, 0, 0, 0)
    p.transform(0, 0, -37, 0, 0, 0)
    

    # Do a cropp (isolation of interest part)
    p.croppcd(-0.7, -0.7, -0.07, 0.7, 0.7, 0.4)

    # Plane detection( Table and objects isolation)
    p.planesegmentation()
    
    # Object Clustering
    p.pcd_clustering()

    # Object isolation and caracterization

    
    # ------------------------------------
    # Visualization
    # ------------------------------------

    #Draw BBox
    entities = []
    bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)

    entities.append(bbox)
    
    # Draw Table Plane
    p.inliers.paint_uniform_color([0.7,0.7,0.7])
    center_table = p.inliers.get_center()
    print('Center of the table: ' + str(center_table))
    entities.append(p.inliers) # Draw only de plane
    
    # Create coordinate system
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0, 0, 0]))
    entities.append(frame)
   


    # Draw table plane + frame + objects
    #entities = np.concatenate((entities, p.objects_to_draw))
#
    #o3d.visualization.draw_geometries(entities,
    #                                         zoom = view['trajectory'][0]['zoom'],
    #                                         front = view['trajectory'][0]['front'],
    #                                         lookat = view['trajectory'][0]['lookat'],
    #                                         up = view['trajectory'][0]['up'])
    #o3d.visualization.draw_geometries(p.objects_to_draw,
    #                                         zoom = view['trajectory'][0]['zoom'],
    #                                         front = view['trajectory'][0]['front'],
    #                                         lookat = view['trajectory'][0]['lookat'],
    #                                         up = view['trajectory'][0]['up'])

   

    #def plot_images():
    ## Defina o número de subplots
    #    n_rows, n_cols = 5, 5
#
    #    # Crie uma figura e adicione subplots
    #    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 7)) # figsize=(width, height)
#
    #    counter = 00
    #    # Itera sobre cada subplot e adiciona uma imagem com título e legenda
    #    for i in range(n_rows):
    #        for j in range(n_cols):
    #            axs[i, j].imshow(np.random.rand(10,10)) # Adiciona uma imagem aleatória
    #            axs[i, j].set_title(f"Objeto {counter:02d}", pad = -10) # pad = -10 para não sobrescrever o título
    #            axs[i, j].axis("off") # Desativa os eixos
    #            #axs[i, j].set_xlabel("Eixo X")
    #            #axs[i, j].set_ylabel("Eixo Y")
    #            counter += 1 # Incrementa o contador
#
    #    # Mostre a figura
    #    fig.suptitle("Objetos em cena") # Título da figura
    #    plt.show()

    #make a more complex window to show 3d objects labels

    app = gui.Application.instance
    app.initialize() # create a open3d app

    w = app.create_window("Detected Objects", 1980, 1080)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    #widget3d.scene.set_background([1,1,1,1])
    material = rendering.Material() #rendering.materialrecord (outras versoes de open3d)
    material.shader = "defaultUnlit"
    material.point_size = 3 * w.scaling
    

    

    for entity_idx, entity in enumerate(p.objects_to_draw):
        widget3d.scene.add_geometry("Entity" + str(entity_idx),entity, material)
        for obj in p.objects_properties:
            l = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.09)), 'Object: ' + str(obj['idx']))
            #volume em (x x y x z) mm
            #l2 = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.06)), 'Volume: ( ' + str(round(obj['x_width']*1000,0)) + 
            #                           ' x ' + str(round(obj['y_width']*1000,0)) + ' x ' + str(round(obj['height']*1000,0)) + ') mm' )
            #area em mm2
            l3 = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.06)), 'Area: (' + str(round(obj['area']* 10000, 0)) + ') cm2')
            #volume em mm3
            l2 = widget3d.add_3d_label(obj['center']+(-0.1,0,((obj['height']/2)+0.04)), 'Volume: (' + str(round(obj['volume']*1000000,0)) + ') cm3')
            #cor label
            l.color = gui.Color(obj['color'][0], obj['color'][1], obj['color'][2],)

    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)
    app.run()
    #app.run_in_thread(plot_images)
    
    
    
    #text to speech ( em gtts) criaçao de arquivo mp3
    def generate_audio(objects_properties):
        message = ""
        for obj in objects_properties:
            message += "O objeto " + str(obj['idx']) + " tem uma área de " + str(round(obj['area'] * 10000, 0)) + " centrímetros quadrados e um volume de " + str(round(obj['volume'] * 100000, 0)) + " centrímetros cúbicos. "

        language = 'pt'
        myobj = gTTS(text=message, lang=language, slow=False)
        myobj.save("Speak_objetos.mp3")

    #    os.system("Speak_objetos.mp3")
    
    #table of info of objects
    #x = PrettyTable() #table default
    x = ColorTable(theme=Themes.OCEAN) #table with colors

    x.field_names = ["Objeto", "Area (cm2)", 'Volume (cm3)']
    x.align = "r"

    for obj in p.objects_properties:
        x.add_row([str(obj['idx']), str(round(obj['area']*10000,0)), str(round(obj['volume']*1000000,0))])

    print('')
    print('Tabela de informações dos objetos:')
    print(x)
    generate_audio(p.objects_properties)


    

    #-------------------------------------------------
    #plot de imagens objetos
    #-------------------------------------------------

    
    # Defina o número de subplots
    n_rows, n_cols = 5, 5

    # Crie uma figura e adicione subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 7)) # figsize=(width, height)

    counter = 00
    # Itera sobre cada subplot e adiciona uma imagem com título e legenda
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].imshow(np.random.rand(10,10)) # Adiciona uma imagem aleatória
            axs[i, j].set_title(f"Objeto {counter:02d}", pad = -10) # pad = -10 para não sobrescrever o título
            axs[i, j].axis("off") # Desativa os eixos
            #axs[i, j].set_xlabel("Eixo X")
            #axs[i, j].set_ylabel("Eixo Y")
            counter += 1 # Incrementa o contador

    # Mostre a figura
    fig.suptitle("Objetos em cena") # Título da figura
    plt.show()

    #text to speech
    engine = pyttsx3.init()
    engine.say("I will speak this text")
    for obj in p.objects_properties:
        message = "O objeto " + str(obj['idx']) + " tem uma área de " + str(round(obj['area'] * 10000, 0)) + " centrímetros quadrados e um volume de " + str(round(obj['volume'] * 100000, 0)) + " centrímetros cúbicos. "
        engine.say(message)

    engine.runAndWait()




    



if __name__ == "__main__":
    main()


