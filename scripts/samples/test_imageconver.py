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
        self.image_sub = rospy.Subscriber("/RGB_image",Image,self.callback,queue_size=1,buff_size=52428800)
        self.image_pub = rospy.Publisher("/maskrcnn_image",Image,queue_size=1)
        self.imageDone = True
     #   cv2.waitKey(1000000)
    def callback(self, data):
        start = time.time()
        if self.imageDone == True:
            self.imageDone = False
            try:
  #          while True:
                cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            except CvBridgeError as e:  
                print(e)
            cv2.waitKey(5000)
            try:
                msg = self.bridge.cv2_to_imgmsg(cv_image,encoding = "bgr8")
                self.image_pub.publish(msg)
            except CvBridgeError as e:  
                print(e)        
            self.imageDone = True
        elapsed =(time.time() - start)
        print ("TIME USED ", elapsed)
def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()

    try:

        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
