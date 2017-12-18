import datetime

from keras import Input, Model, optimizers
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Lambda, Conv2DTranspose, Add, UpSampling2D

import os, sys, random, math
import numpy as np
import skimage.io
import skimage.transform
import skimage.util
from skimage.color import gray2rgb, rgb2gray, rgb2lab, lab2rgb
import matplotlib
import matplotlib.pyplot as plt

import coco, utils, visualize
import model as modellib
import keras.backend as K
from keras.metrics import mean_squared_error


ROOT_DIR = os.getcwd()
MASK_LOGS_DIR = os.path.join(ROOT_DIR, 'Mask_auto_color_batchnorm_logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_batchnorm_coco.h5')

TB_LOG_DIR = os.path.join(ROOT_DIR, 'auto_color_tb_bc_logs')
TESTING_RESULT_DIR = os.path.join(ROOT_DIR, 'auto_color_mrcnn_bc_test')

if os.getlogin() == 'yongjiang':
    IMAGE_DIR = os.path.join(ROOT_DIR, 'images')
    TRAINING_DIR = '/Users/yongjiang/PycharmProjects/tensorflow/COCOsubset/train2017/'
    TESTING_DIR = '/Users/yongjiang/PycharmProjects/tensorflow/COCOsubset/train2017/'
    BATCH_SIZE = 2
    SHOW = True

elif os.getlogin() == 'ubuntu' :
    IMAGE_DIR = '/home/ubuntu/411csc/train2017/train2017'
    TRAINING_DIR = '/home/ubuntu/411csc/train2017/train2017'
    TESTING_DIR = os.path.join(ROOT_DIR, 'images')
    BATCH_SIZE = 10
    SHOW = False

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
    mask_model.keras_model.trainable = False

    class_names = CLASS_NAMES

    file_names = next(os.walk(IMAGE_DIR))[2]
    random.shuffle(file_names)
    
    if SHOW is True:
        test_file_names = next(os.walk(TESTING_DIR))[2][0:1]
    else:
        test_file_names = next(os.walk(TESTING_DIR))[2][0:10]

    def log_auto_color(s):
        try:
            log_file = open("loss_hist_bc.txt", 'a')
            log_file.write('{}\n'.format(s))
            # print("{}: {}".format(str(datetime.now()), s)) # For debugging
        except:
            None
        finally:
            log_file.close()
    
    def report_loss(filenames = test_file_names):
        X_batch, Y_batch, _ = generate_training_datum(filenames, image_dir=TESTING_DIR)
        loss = model.evaluate(X_batch, Y_batch)
        print("Loss(bachnorm): {}".format(loss))
        log_auto_color(loss)
    
    def colorize(filename='000000000813.jpg'):
        """Colorize One picture"""
        X_batch, Y_batch, images = generate_training_datum([filename], image_dir=TESTING_DIR)
        
        # plt.imshow(images[0])
        # plt.show()
        
        preds = model.predict(X_batch)
        # print("predict shape: {}".format(preds.shape))
        
        lab = rgb2lab(images[0])
        pred_image = np.zeros(lab.shape)
        pred_image[:, :, 0] = lab[:, :, 0]
        pred_image[:, :, 1:] = preds[0] * 128
        
        # pred_image = np.concatenate((images[0], lab2rgb(pred_image)), axis=1) # Demage orignal image
        pred_image = lab2rgb(pred_image)
        # print(preds)
        
        if SHOW is True:
            plt.imshow(pred_image)
            plt.show()
        return pred_image

    def get_feature_map(images):
        """ Get the feature map from the trained mask_rcnn """
        result = mask_model.run_graph(images, [
            ('P2', mask_model.keras_model.get_layer('fpn_p2').output), # -> shape: (2, 256, 256, 256)
            ('P3', mask_model.keras_model.get_layer('fpn_p3').output), # -> shape: (2, 128, 128, 256)
            ('P4', mask_model.keras_model.get_layer('fpn_p4').output), # -> shape: (2, 64, 64, 256)
            ('P5', mask_model.keras_model.get_layer('fpn_p5').output), # -> shape: (2, 32, 32, 256)
        ])
        return result

    def generate_training_datum(filenames, image_dir=IMAGE_DIR):
        images = []
        grayscaled_rgbs = []
        Y_batch = []
        for filename in filenames:
            image = skimage.io.imread(os.path.join(image_dir, filename))
            image, _, _, _ = utils.resize_image(image, min_dim=config.IMAGE_MAX_DIM)
            
            try:
                image = image[:config.IMAGE_SHAPE[0], :config.IMAGE_SHAPE[1], :]
            except IndexError:
                continue
                
            images.append(image)
            lab = rgb2lab(image)
            
            grayscaled_rgb = gray2rgb(rgb2gray(image))
            grayscaled_rgbs.append(grayscaled_rgb)
        
            Y_batch.append(lab[:, :, 1:]/128)
            
        feature_maps = get_feature_map(grayscaled_rgbs)
        # print(feature_maps['P2'].shape) # -> (batch_size, pool_size, pool_size, filter_num)
        
        grayscaled_rgbs = np.asarray(grayscaled_rgbs)
        Y_batch = np.asarray(Y_batch)
        # print(grayscaled_rgbs.shape) # -> (batch_size, height, width, channels)
        
       
        return feature_maps, Y_batch, images

        
    # generate_training_datum(file_names[0:2])
    
    # ========= Building the network ========= #
    # Input: https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
    P5 = Input(shape=(32, 32, 256,),   name='P5')
    P4 = Input(shape=(64, 64, 256,),   name='P4')
    P3 = Input(shape=(128, 128, 256,), name='P3')
    P2 = Input(shape=(256, 256, 256,), name='P2')
    
    # Decode
    decode_p5 = Conv2D(128, (3, 3), activation='relu', padding='same')(P5)
    decode_p5 = UpSampling2D((2, 2))(decode_p5)
    decode_p4 = Conv2D(128, (1,1), activation='relu', padding='same')(P4)
    decode_p4_5 = Add()([decode_p5, decode_p4])
    
    decode2_p4_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(decode_p4_5)
    decode2_p4_5 = UpSampling2D((2,2))(decode2_p4_5)
    decode2_p3 = Conv2D(64, (3, 3), activation='relu', padding='same')(P3)
    decode2_p3_4_5 = Add()([decode2_p4_5, decode2_p3])
    
    decode3_p345 = Conv2D(32, (3, 3), activation='relu', padding='same')(decode2_p3_4_5)
    decode3_p345 = UpSampling2D((2, 2))(decode3_p345)
    decode3_p2 = Conv2D(32, (1,1), activation='relu', padding='same')(P2)
    decode3_p2345 = Add()([decode3_p345, decode3_p2])
    
    decode_out = Conv2D(16, (3, 3), activation='relu', padding='same')(decode3_p2345)
    decode_out = UpSampling2D((2, 2))(decode_out)
    decode_out = Conv2D(4, (3, 3), activation='relu', padding='same')(decode_out)
    decode_out = UpSampling2D((2, 2))(decode_out)
    decode_out = Conv2D(2, (3, 3), activation='tanh', padding='same')(decode_out)
    
    # build
    tensorboard = TensorBoard(log_dir=TB_LOG_DIR)
    model = Model(inputs=[P5, P4, P3, P2], outputs=decode_out)
    
    if os.path.isfile('auto_color_batch_norm.h5'):
        print('Found weights')
        model.load_weights('auto_color_batch_norm.h5')
        
    sgd = optimizers.SGD(lr=0.05, momentum=0.1, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss='mse')
    
    
    # ========= Training =========== #
    batch_size = BATCH_SIZE
    for i in range(int(len(file_names)/batch_size-1)):
    # for i in range(30):
        print('(batchnorm) Training on batch {}'.format(i))
        X_batch, Y_batch, _ = generate_training_datum(file_names[ i*batch_size : (i+1)*batch_size ])
        model.train_on_batch(X_batch, Y_batch)
        
        if SHOW is True:
            colored = colorize()

        # color_files = random.choice(test_file_names)
        # report_loss()

        if i % 10 == 0:
            report_loss()
            colored = colorize()
            skimage.io.imsave(os.path.join(TESTING_RESULT_DIR, '{}_test_batchnorm_'.format(i) + "00000813.jpg"),arr=colored)
        
        if i % 500 == 499:
            model.save_weights("{}_color_batchnorm_mrcnn.h5".format(i))
            
    # ===== Store Model ===== #
    # Save model
    model_json = model.to_json()
    with open("batchnorm_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("auto_color_batchnorm_final.h5")


if __name__ == '__main__':
    main()