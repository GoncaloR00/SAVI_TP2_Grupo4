#!/usr/bin/env python3

depth_topic = 'camera/depth_registered/image_raw'
rgb_topic = 'camera/rgb/image_raw'

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(depth_topic,Image,self.callback)
    self.counter = 0

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    except CvBridgeError as e:
      print(e)

    (rows,cols) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)
    string_number = str(self.counter)
    path = './depth/' + string_number + '.png'
    # depth_array = np.array(cv_image, dtype=np.float32)
    # cv2.imwrite(path, cv_image)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_colormap)
    # depth_array = depth_array.astype(np.uint16)
    # cv2.imwrite("depth_img.png", depth_array)
    print(string_number)
    cv2.waitKey(3)
    self.counter += 1 

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)