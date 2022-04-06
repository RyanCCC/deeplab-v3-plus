from pickle import TRUE


num_classes = 19
backbone    = "xception"
model_path  = "model/deeplabv3tf2_xception.h5"
dataset_path = './Cityscapes'

LABEL_COLORS = [(128, 64, 128), (231, 35, 244), (69, 69, 69)
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
                ,(119, 10, 32)]
                # 18 = bicycle

downsample_factor = 16
input_shape = [512, 512]
START_EPOCH = 0
Freeze_Epoch = 100
FREEZE_BATCHSIZE = 8
FREEZE_LEARNING_RATE = 5e-4
UNFREEZE_EPOCH = 200
UNFREEZE_BATCHSIZE = 4
UNFREEZE_LEARNING_RATE = 5e-5
FREEZE_TRAIN = True
DICE_LOSS = False