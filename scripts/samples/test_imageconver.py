#!/usr/bin/env python3
import roslib
import cv2
import numpy as np
#roslib.load_manifest('my_package')
import rospy
import os
import sys
import random
import math
#import skimage.io
import time
#import utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
#from mrcnn import utils
#import mrcnn.model as modellib
#from coco import coco

class image_converter:  

    def __init__(self):

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/RGB_image",Image,self.callback)
        cv2.waitKey(1000000)
    def callback(self, data):
        currentframe = 0
        try:
            while True:

                cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')


        except CvBridgeError as e:  
            print(e)
def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        cv2.waitKey(1000)
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
