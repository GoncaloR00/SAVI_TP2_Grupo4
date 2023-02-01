#!/usr/bin/env python3
import pyttsx3
import rospy
from savi_tp2.msg import images
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import math
# ---------------------------------------------------
#   RosTopic to subscribe
# ---------------------------------------------------
topic_images = '/tp2_savi/images'

def callback(message):
    newVoiceRate = 120
    engine = pyttsx3.init()
    engine.setProperty('rate',newVoiceRate)
    text = "There are "
    names_list = []
    for class_str in message.classes:
        names_list.append(class_str.data)
    classes_list = [*set(names_list)]
    for idx, class_name in enumerate(classes_list):
        num = names_list.count(class_name)
        if idx < len(classes_list) - 1:
            text = text + str(num) + " " + class_name + ", "
        else:
            text = text + "and " + str(num) + " " + class_name + "."
    engine.say(text)
    engine.runAndWait()

def listener():
    rospy.init_node('speech')
    rospy.Subscriber(topic_images, images, callback)
    rospy.spin()

if __name__ == "__main__":
    listener()