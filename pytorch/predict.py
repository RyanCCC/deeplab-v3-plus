from deeplab.deeplab import *
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as tr
from dataloaders.utils import decode_segmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def transform(image):
    return tr.Compose([
        tr.Resize(513),
        tr.CenterCrop(513),
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

if __name__ == '__main__':
    # 加载模型
    model = DeepLab(backbone='resnet', output_stride=16)
    state_dict = torch.load('./checkpoints/checkpoint.pth')
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    model.to(device)

    torch.set_grad_enabled(False)
    # 图像预处理
    image_path = './images/2007_000720.jpg'
    img = Image.open(image_path)
    inputs = transform(img).to(device)
    # 推理
    output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
    pred = np.argmax(output, axis=0)
    # 后处理：TODO
    decode_segmap(pred, dataset="pascal", plot=True)