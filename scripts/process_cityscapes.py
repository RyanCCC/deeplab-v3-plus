import os
from tqdm import tqdm
from glob import glob
from shutil import copy


leftImg_pattern = './Cityscapes/leftImg8bit/val/*/*'
gtFine_pattern  = './Cityscapes/gtFine/val/*/*gtFine_labelTrainIds.png'

JPEGImage_savepath = './Cityscapes/JPEGImage'
Label_savepath = './Cityscapes/Label'

if not os.path.exists(JPEGImage_savepath):
    os.mkdir(JPEGImage_savepath)

if not os.path.exists(Label_savepath):
    os.mkdir(Label_savepath)

JPEGImages = glob(leftImg_pattern)
labels = glob(gtFine_pattern)

for i in tqdm(range(len(JPEGImages))):
    jpeg_basename =os.path.basename(JPEGImages[i])
    label_basename = os.path.basename(labels[i])

    dest_jpeg = os.path.join(JPEGImage_savepath, jpeg_basename)
    dest_label = os.path.join(Label_savepath, jpeg_basename)

    copy(JPEGImages[i], dest_jpeg)
    copy(labels[i], dest_label)

print('finish')