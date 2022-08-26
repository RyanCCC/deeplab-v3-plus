from deeplab.deeplab import *
import torch
from PIL import Image
from utils.tools import *
import copy
import numpy as np
import cv2

if __name__ == '__main__':
    # 加载模型
    model = DeepLab(backbone='resnet', output_stride=16)
    state_dict = torch.load('./checkpoints/checkpoint.pth')
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    input_shape = (513, 513)
    # 图像预处理
    image_path = './images/2007_000720.jpg'
    img = Image.open(image_path)
    ori_image = copy.deepcopy(img)
    orininal_h  = np.array(img).shape[0]
    orininal_w  = np.array(img).shape[1]
    image_data, nw, nh  = resize_image(img, input_shape)
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        pr = model(images)[0]
    # 后处理：TODO
    pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
    pr = pr[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh), \
                    int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
    # pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
    pr = pr.argmax(axis=-1)
    print(output.size())