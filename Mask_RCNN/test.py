import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage.color import rgb2gray, gray2rgb


def print_loss():
    yuv_mse_loss =[0.16464535892009735, 0.04048551991581917, 0.007973773404955864, 0.004171219654381275, 0.003717037383466959, 0.003656623186543584, 0.0036587808281183243, 0.0036763623356819153, 0.0036677648313343525, 0.003679195884615183, 0.0036708880215883255, 0.0036920742131769657, 0.003696648869663477, 0.0036708880215883255,]

    yuv_mlog_loss =  [0.19353027641773224, 0.18188926577568054,  0.17299506068229675,  0.16056784987449646,  0.14331600069999695,  0.11890523135662079,  0.0868103951215744,  0.05139455199241638,  0.024101680144667625,  0.009437652304768562,  0.0038455177564173937,  0.002203818177804351,  0.0017246352508664131,  0.001419494510628283]

    yuv_mabs_loss =  [0.8133233189582825, 0.7665883898735046, 0.6963841915130615, 0.584185779094696, 0.3927682042121887, 0.11951277405023575, 0.04182317107915878, 0.04225069284439087, 0.04243633896112442, 0.04247504472732544, 0.042627450078725815, 0.04180024191737175, 0.04204896092414856, 0.042173899710178375]

    
    lab_mse_loss = []
    lab_mlog_loss = []
    lab_mabs_loss = []
    
    
    batch_steps = range(14)
    
    plt.plot(batch_steps, yuv_mse_loss)
    plt.plot(batch_steps, yuv_mlog_loss)
    plt.plot(batch_steps, yuv_mabs_loss)
    plt.legend(['mean_square_error', 'mean_squared_logarithmic_error', 'mean_absolute_error'], loc='upper left')
    
    plt.show()
    


def to_grayscale():
    image = skimage.io.imread('images/000000000643.jpg')
    grayscale = gray2rgb(rgb2gray(image))
    skimage.io.imsave('demo_gray.jpg', grayscale)
    



if __name__ == '__main__':
    # print_loss()
    to_grayscale()