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
#import utils
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
 #       image[:, :, n] = np.where(
 #           mask == 1,
 #           image[:, :, n] *(1 - alpha) + alpha * c,
 #           image[:, :, n]
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

 
if __name__=='__main__':
    import os
    import sys
    import random
    import math

    import time
    import utils
    #import model as modellib
    
    
    ROOT_DIR = os.path.abspath("")
    sys.path.append(ROOT_DIR)
 
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        #COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    COCO_MODEL_PATH = ("/media/tx2/Drive/wuzida/catkin_ws/src/maskRCNN/scripts/mask_rcnn_coco.h5")
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
    file_names = next(os.walk(IMAGE_DIR))[2]
    while True:
        for i in range(len(file_names)):
            image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
    #capture=cv2.VideoCapture(0)
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    
       # ret,frame=capture.read()
            results=model.detect([image],verbose=0)
            r=results[0]
        
        
            image=display_instances(
              image,r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores']
            )
            saveImageName = './maskimage'+ i;
            cv2.imshow('image', image)
            cv2.waitKey(1000)
            cv2.imwrite(saveImageName,image)
            cv2.destroyAllWindows()
       
  #  capture.release()

