import os
import sys
import onnx
from onnxconverter_common import float16
import glob


def convert(src, dst):
    model = onnx.load(src)
    model_fp16 = float16.convert_float_to_float16(model, op_block_list=["RandomNormalLike"])
    onnx.save(model_fp16, dst)

def convert_all(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for onnx_f32 in glob.glob(os.path.join(src_dir, "*.onnx")):
        onnx_f16 = os.path.join(dst_dir, os.path.basename(onnx_f32))
        convert(onnx_f32, onnx_f16)

if __name__ == '__main__':
    convert_all('onnx/test', 'onnx_f16/test')
