import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import config
from utils.utils import cvtColor, preprocess_input, resize_image
import copy
import colorsys
import onnxruntime

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.keras.models.load_model('./cityscapes_model', custom_objects={'tf':tf})

def get_colors(num_classes):
    if num_classes <= 21:
        colors = [ (128, 64, 128), (231, 35, 244), (69, 69, 69)
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
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors
        

def inference_onnx(image,onnx_path = './road.onnx', input_shape = config.input_shape, num_classes=config.num_classes, blend=True):
    image = cvtColor(image)
    old_image = copy.deepcopy(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    image_data, nw, nh  = resize_image(image, (input_shape[1], input_shape[0]))
    image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
    colors = get_colors(num_classes)
    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_data})
    pred = np.squeeze(outputs[0], axis = 0)
    pred = pred.argmax(axis=-1).reshape([input_shape[0],input_shape[1]])
    pred = pred[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh), \
                int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
    if blend:
        seg_img = np.zeros((np.shape(pred)[0], np.shape(pred)[1], 3))
        for c in range(num_classes):
            seg_img[:,:,0] += ((pred[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pred[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pred[:,: ] == c )*( colors[c][2] )).astype('uint8')
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        image = Image.blend(old_image,image,0.7)
    # get miou
    else:
        image = Image.fromarray(np.uint8(pred)).resize((orininal_w, orininal_h), Image.NEAREST)

    return image

def pred_func(image,model=model, input_shape = config.input_shape, num_classes=config.num_classes, blend=True):
    image = cvtColor(image)
    old_image = copy.deepcopy(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    image_data, nw, nh  = resize_image(image, (input_shape[1], input_shape[0]))
    image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
    colors = get_colors(num_classes)
    @tf.function
    def get_pred(image_data):
        pr = model(image_data, training=False)
        return pr
    
    pred = get_pred(image_data)[0].numpy()
    pred = pred.argmax(axis=-1).reshape([input_shape[0],input_shape[1]])
    pred = pred[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh), \
                int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
    if blend:
        seg_img = np.zeros((np.shape(pred)[0], np.shape(pred)[1], 3))
        for c in range(num_classes):
            seg_img[:,:,0] += ((pred[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pred[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pred[:,: ] == c )*( colors[c][2] )).astype('uint8')
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        image = Image.blend(old_image,image,0.7)
    # get miou
    else:
        image = Image.fromarray(np.uint8(pred)).resize((orininal_w, orininal_h), Image.NEAREST)

    return image
        
    

if __name__ == "__main__":
    
    image = Image.open('./testimg/20220823115704.jpg')

    image = pred_func(image)
    image.show()
    # 视频语义分割测试
    if False:
        video_path      = 0
        video_save_path = ""
        video_fps       = 25.0
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(deeplab.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()