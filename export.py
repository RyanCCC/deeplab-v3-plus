import colorsys
import copy
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from nets.deeplab import Deeplabv3
import config as sys_config


def ExportDeeplabV3Model():
    params = {
        "model_path"        : sys_config.MODEL_WEIGHTS,
        "num_classes"       : sys_config.num_classes,
        "backbone"          : sys_config.backbone,
        "input_shape"       : sys_config.input_shape,
        "downsample_factor" : sys_config.downsample_factor,
        "blend"             : True,
    }
    model = Deeplabv3(input_shape=[params['input_shape'][0], params['input_shape'][1], 3], num_classes= params['num_classes'],
                               backbone= params['backbone'], downsample_factor= params['downsample_factor'])

    model.load_weights(params['model_path'])
        # 导出model 
    model.save('./cityscapes_model')
    print('success export model')

ExportDeeplabV3Model()