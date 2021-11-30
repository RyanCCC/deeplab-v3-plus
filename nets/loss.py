import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import log


def dice_loss_with_CE(beta=1, smooth = 1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE

def CE():
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE

# 添加focal loss
def focal_loss(gamma=2, beta=0.5):
    def focal_loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = -y_true[..., :-1]*K.log(y_pred)
        pt = K.sum(pt, axis=-1)
        pt = tf.exp(pt)
        if beta is not  None:
            pt = pt*beta
        focal_loss_value = -((1 - pt) ** gamma) * pt
        focal_loss_value = K.mean(focal_loss_value)
        return focal_loss_value
    return focal_loss_fn