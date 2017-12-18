from keras import Input, Model, optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Lambda

import os, sys, random, math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco, utils, visualize
import model as modellib
import keras.backend as K
import theano

ROOT_DIR = os.getcwd()
MASK_LOGS_DIR = os.path.join(ROOT_DIR, 'Mask_logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

IMAGE_DIR = os.path.join(ROOT_DIR, 'images')

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

# Configuration of Mask Model

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def main():
    config = InferenceConfig()
    # config.display()
    
    
    # Create Mask Model
    mask_model = modellib.MaskRCNN(mode='inference', model_dir=MASK_LOGS_DIR, config=config)
    mask_model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    # Freeze model
    mask_model.keras_model.trainable=False
    
    class_names = CLASS_NAMES
    
    
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    
    # roi_align_mask_0 = mask_model.run_graph([image], [
    #     ('detections', mask_model.keras_model.get_layer('mrcnn_detection').output),
    #     ('roi_align_mask_0', mask_model.keras_model.get_layer('roi_align_mask').output),
    # ])
    
    def get_rois_x(image):
        """
        
        :param image:
        :return:
            roi_align_mask: Tensor (1, 100, 14, 14, 256) seems to be same, since the inner model.input is same
            mrcnn_mask: Tensor (1, 100, 28, 28, 81)
        """
        result =  mask_model.run_graph([image], [
            # ('detections', mask_model.keras_model.get_layer('mrcnn_detection').output),
            ('roi_align_mask_0', mask_model.keras_model.get_layer('roi_align_mask').output),
            ('mrcnn_mask_0', mask_model.keras_model.get_layer('mrcnn_mask').output),
        ])
        # return result['roi_align_mask_0']
        return result
    # Test get_rois_x
    # print(get_rois_x(image)['mrcnn_mask_0'].shape)
    
    def get_feature_map(image):
        """ Get the feature map from the trained mask_rcnn """
        result = mask_model.run_graph([image], [
            ('P2', mask_model.keras_model.get_layer('fpn_p2').output),
            ('P3', mask_model.keras_model.get_layer('fpn_p3').output),
            ('P4', mask_model.keras_model.get_layer('fpn_p4').output),
            ('P5', mask_model.keras_model.get_layer('fpn_p5').output),
        ])
        return result
    
    for P in get_feature_map(image):
        print('Shape: {}'.format(P.shape))
    
    # input_image = Input(shape=(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3,), name="image_input")
    #
    # model = Sequential()

    
    
    
    # model = Model(inputs=input_image, outputs=)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='mean_squred_error')
    
    
    

if __name__ == '__main__':
    main()