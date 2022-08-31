import numpy as np
import tensorflow as tf
from PIL import Image
from nets.deeplab import Deeplabv3
import config as sys_config
import argparse
import tf2onnx

def parser_opt():
    parser = argparse.ArgumentParser(description='deeplab模型导出')
    parser.add_argument('--flag', action='store_true', help='True:Tensoflow model, False:Tensorflow weights')
    parser.add_argument('--model_path', type=str, help='导出的模型或权重', default='')
    parser.add_argument('--saved_pb', action='store_true', help='是否保存pb模型')
    parser.add_argument('--saved_pb_dir', type=str, help='保存pb格式的模型')
    parser.add_argument('--save_onnx', type=str, help='save onnx model name', required=True, default='')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--num_class', type=int, required=True, help='类别数')
    parser.add_argument('--backbone', choices=['mobilenet', 'xception'], default='xception',help='骨干网络：backbone, mobilenet')
    return parser


def main(args):
    num_class = args.num_class
    onnx_save_path = args.save_onnx
    opset = args.opset
    saved_model = args.model_path
    backbone = args.backbone
    if args.flag:
        '''
        从tensorflow模型中导出onnx模型
        '''
        assert len(saved_model) > 0, 'saved_model cannot be none or empty.'
        deeplab_model = tf.keras.models.load_model(saved_model)
        model_proto, _ = tf2onnx.convert.from_keras(deeplab_model, opset=opset, output_path=onnx_save_path)
        output_names = [n.name for n in model_proto.graph.output]
        print(output_names)
    else:
        model = Deeplabv3(input_shape=[sys_config.input_shape[0], sys_config.input_shape[1], 3], num_classes= num_class,
                               backbone= backbone, downsample_factor= sys_config.downsample_factor)
        model.load_weights(saved_model)
        save_pb = args.saved_pb
        if save_pb:
            save_name = args.saved_pb_dir
            assert len(save_name) > 0, 'save_name cannot be none or empty.'
            model.save(save_name, save_format='tf')
        model_proto, _ = tf2onnx.convert.from_keras(model, opset=opset, output_path=onnx_save_path)
        output_names = [n.name for n in model_proto.graph.output]
        print(f'Model output names: ',output_names)


if __name__ == '__main__':
    parser = parser_opt()
    args = parser.parse_args()
    main(args=args)