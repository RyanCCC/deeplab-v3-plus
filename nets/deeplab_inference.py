import colorsys
import copy
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image
import config as sys_config


class DeeplabV3(object):
    _defaults = {
        "model_path"        : 'model/model_1.h5',
        "num_classes"       : sys_config.num_classes,
        "backbone"          : sys_config.backbone,
        "input_shape"       : [512, 512],
        "downsample_factor" : 16,
        "blend"             : True,
    }
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.__dict__.update(self._defaults)
        if self.num_classes <= 21:
            self.colors = [ (128, 64, 128), (231, 35, 244), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32), (128, 192, 0), (0, 64, 128)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):
        self.model = Deeplabv3([self.input_shape[0], self.input_shape[1], 3], self.num_classes,
                                backbone = self.backbone, downsample_factor = self.downsample_factor)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))
        
    @tf.function
    def get_pred(self, image_data):
        pr = self.model(image_data, training=False)
        return pr

    def detect_image(self, image):
        image       = cvtColor(image)
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)


        pr = self.get_pred(image_data)[0].numpy()
        pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))

        if self.blend:
            image = Image.blend(old_img,image,0.7)

        return image

    def get_FPS(self, image, test_interval):
        image       = cvtColor(image)
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        pr = self.get_pred(image_data)[0].numpy()
        pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
                
        t1 = time.time()
        for _ in range(test_interval):
            pr = self.get_pred(image_data)[0].numpy()
            pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
    def get_miou_png(self, image):
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        pr = self.get_pred(image_data)[0].numpy()
        pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)
        return image