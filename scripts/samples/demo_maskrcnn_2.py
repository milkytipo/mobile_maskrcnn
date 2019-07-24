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
import skimage.io
import time
import utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from mrcnn import utils
import mrcnn.model as modellib
from coco import coco

def random_colors(N):
    np.random.seed(1)
    colors=[tuple(255*np.random.rand(3)) for _ in range(N)]
    return colors
 
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n],
            image[0, 0, n]
        )
    return image
 
def display_instances(image,boxes,masks,ids,names,scores):
    n_instances=boxes.shape[0]
    if not n_instances:
        print('No instances to display')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    
    colors=random_colors(n_instances)
    height, width = image.shape[:2]
    
    for i,color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        
 #       y1,x1,y2,x2=boxes[i]
 #       mask=masks[:,:,i]
 #       image=apply_mask(image,mask,color)
 #       image=cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        
 #       label=names[ids[i]]
 #       score=scores[i] if scores is not None else None
        
 #       caption='{}{:.2f}'.format(label,score) if score else label
 #       image=cv2.putText(
  #          image,caption,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.7,color,2
#        )
        label=names[ids[i]]
        if label == 'keyboard':
            y1,x1,y2,x2=boxes[i]
            mask=masks[:,:,i]
            image=apply_mask(image,mask,color)
            image=cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        
            score=scores[i] if scores is not None else None
        
            caption='{}{:.2f}'.format(label,score) if score else label
            image=cv2.putText(
            image,caption,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.7,color,2
        )
        
    return image


class image_converter:
 
    def __init__(self):
        self.image_sub = rospy.Subscriber("/RGB_image",Image,self.callback,queue_size=1)
        self.image_pub = rospy.Publisher("/maskrcnn_image",Image,queue_size=1)
        ROOT_DIR = os.path.abspath("") # this path is not the .py path, its the command terminal path!
        sys.path.append(ROOT_DIR)
 
        sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

        if not os.path.exists(COCO_MODEL_PATH):
            print('cannot find coco_model')
            utils.download_trained_weights(COCO_MODEL_PATH)

        IMAGE_DIR = os.path.join(ROOT_DIR, "images2")

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
 
        config = InferenceConfig()
        config.display()
    
        model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_DIR, config=config
        )
        print("rcnn network ready")
 
        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        print("rcnn weights load ready")
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
        bridge = CvBridge()
        file_names = os.listdir(IMAGE_DIR)

        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
    
            for i in range(len(file_names)):
                start = time.time()
                cv_image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))

                results=model.detect([cv_image],verbose=1)
                r=results[0]

                cv_image=display_instances(
                    cv_image,r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                )
                msg = bridge.cv2_to_imgmsg(cv_image,encoding = "bgr8")
                elapsed =(time.time() - start)
                print ("TIME USED ", elapsed)
                self.image_pub.publish(msg)
                rate.sleep()

    def callback(self,ros_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
          #  np_arr = np.fromstring(ros_data.data, np.uint8)
          #  cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)
 
        (rows,cols,channels) = cv_image.shape

        cv2.imshow("Image window", cv_image)

       # results=model.detect([cv_image],verbose=1)
       # r=results[0]

       # cv_image=display_instances(
       #     cv_image,r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
       # )
        
        cv2.imshow('frame',cv_image)

#        msg = CompressedImage()
#        msg.header.stamp = rospy.Time.now()
#        msg.format = "jpeg"
#        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
        # Publish new image
#        self.image_pub.publish(msg)
 #           self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
 #       except CvBridgeError as e:
 #           print(e)


def main(args): 
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main(sys.argv)
