#!/usr/bin/env python3
import rospy
from savi_tp2.msg import images
from cv_bridge import CvBridge
from std_msgs.msg import String
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------
#   RosTopic to subscribe
# ---------------------------------------------------
topic_images = '/tp2_savi/images'

def callback(message):
    bridge = CvBridge()
    images_list = []
    for image in message.images:
        images_list.append(bridge.imgmsg_to_cv2(image, "rgb8"))
    if len(images_list) > 0:
        size = int(math.sqrt(len(images_list)) + (0 if math.sqrt(len(images_list))% 1 == 0 else 1))
        fig, axs = plt.subplots(size, size)
        fig.tight_layout()
        fig.canvas.set_window_title('Images and their Labels')
        for i in range(size):
            for j in range(size):
                if (i*size + j) < len(images_list):
                    axs[i, j].imshow(images_list[i*size + j])
                    axs[i, j].set_title(message.classes[i*size + j].data)
                    axs[i, j].axis('off')
        plt.show()

def listener():
    rospy.init_node('image_plot')
    rospy.Subscriber(topic_images, images, callback)
    rospy.spin()

if __name__ == "__main__":
    listener()

#!/usr/bin/env python3
# import rospy
# from savi_tp2.msg import images
# from cv_bridge import CvBridge
# from std_msgs.msg import String

# # Para demonstrar
# import cv2

# # ---------------------------------------------------
# #   RosTopic to subscribe
# # ---------------------------------------------------
# topic_images = '/tp2_savi/images'

# def callback(message):
#     bridge = CvBridge()
#     images_list = []
#     # print(message)
#     for image in message.images:
#         images_list.append(bridge.imgmsg_to_cv2(image, "rgb8"))
#     if len(images_list) > 0:
#         for idx, image in enumerate(images_list):
#             cv2.imshow('Imagem ' + str(idx) + ': '+ message.classes[idx].data, image)
#         cv2.waitKey(0)


# def listener():
#     rospy.init_node('image_plot')
#     rospy.Subscriber(topic_images, images, callback)
#     rospy.spin()

# if __name__ == "__main__":
#     listener()