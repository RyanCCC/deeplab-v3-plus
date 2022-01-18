import os
from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.deeplab import Deeplabv3
from nets.loss import CE, dice_loss_with_CE
from utils.callbacks import (ExponentDecayScheduler,
                             ModelCheckpoint)
from utils.dataloader import DeeplabDataset
from utils.utils_fit import fit_one_epoch
from utils.utils_metrics import Iou_score, f_score, fast_hist
import config as sys_config

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    

if __name__ == "__main__":

    eager       = False
    num_classes = sys_config.num_classes
    #   mobilenet、xception 
    backbone    = sys_config.backbone
    model_path  = sys_config.model_path
    #   下采样的倍数8、16 
    #   8要求更大的显存
    downsample_factor   = 16
    input_shape         = [512, 512]
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 5e-4
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 5e-5

    dataset_path  = sys_config.dataset_path
    dice_loss       = False
    Freeze_Train    = True
    num_workers     = 0


    model = Deeplabv3([input_shape[0], input_shape[1], 3], num_classes, backbone = backbone, downsample_factor = downsample_factor)

    model.load_weights(model_path, by_name=True,skip_mismatch=True)
    with open(os.path.join(dataset_path, "train.txt"),"r") as f:
        train_lines = f.readlines()

    with open(os.path.join(dataset_path, "val.txt"),"r") as f:
        val_lines = f.readlines()

    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint_path = os.path.join('logs', 'model.h5')
    checkpoint      = ModelCheckpoint(checkpoint_path,
                        monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.92, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


    loss            = dice_loss_with_CE() if dice_loss else CE()
    if backbone=="mobilenet":
        freeze_layers = 146
    else:
        freeze_layers = 358
    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        train_dataloader    = DeeplabDataset(train_lines, input_shape, batch_size, num_classes, True, dataset_path, 'JPEGImage', 'Label')
        val_dataloader      = DeeplabDataset(val_lines, input_shape, batch_size, num_classes, False, dataset_path, 'JPEGImage', 'Label')

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataloader()), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader()), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.92, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, f_score())

        else:
            model.compile(loss = loss,
                    optimizer = Adam(lr=lr),
                    metrics = [f_score()])

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers != 0 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping]
            )

    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        train_dataloader    = DeeplabDataset(train_lines, input_shape, batch_size, num_classes, True, dataset_path, 'JPEGImage', 'Label')
        val_dataloader      = DeeplabDataset(val_lines, input_shape, batch_size, num_classes, False, dataset_path, 'JPEGImage', 'Label')

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.92, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, f_score())

        else:
            model.compile(loss = loss,
                    optimizer = Adam(lr=lr),
                    metrics = [f_score()])

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers != 0 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping]
            )