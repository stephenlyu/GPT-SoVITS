import onnxruntime as ort
import os
import numpy as np
import torch
import sys
import time

class T2SModelOnnxRT:
    def __init__(self, onnx_root, is_half=True, prefix=None):
        print('T2SModelOnnxRT.__init__')
        self.onnx_root = onnx_root
        self.prefix = prefix
        self.is_half = is_half
        self.EOS = 1024
        self.init_sessions()

    def init_sessions(self):        
        prefix = self.prefix
        if prefix is None:
            prefix = os.path.basename(self.onnx_root)
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 1
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        self.encoder = ort.InferenceSession(os.path.join(self.onnx_root, '%s_t2s_encoder.onnx' % prefix), 
                                            sess_opt, providers=providers)
        self.fsdec = ort.InferenceSession(os.path.join(self.onnx_root, '%s_t2s_fsdec.onnx' % prefix), 
                                          sess_opt, providers=providers)
        self.sdec = ort.InferenceSession(os.path.join(self.onnx_root, '%s_t2s_sdec.onnx' % prefix), 
                                         sess_opt, providers=providers)

    def encode(self, all_phoneme_ids, bert):
        binding = self.encoder.io_binding()

        binding.bind_input(
            name='all_phoneme_ids',
            device_type='cuda',
            device_id=0,
            element_type=np.int64,
            shape=tuple(all_phoneme_ids.shape),
            buffer_ptr=all_phoneme_ids.data_ptr())
        binding.bind_input(
            name='bert',
            device_type='cuda',
            device_id=0,
            element_type=np.float16 if self.is_half else np.float32,
            shape=tuple(bert.shape),
            buffer_ptr=bert.data_ptr())
        binding.bind_output('x')
        self.encoder.run_with_iobinding(binding)

        x = binding.get_outputs()[0]
        return x

    def first_stage_decode(self, x, prompts):
        binding = self.fsdec.io_binding()
        binding.bind_input(name='x', 
                           device_type=x.device_name(), 
                           device_id=0, 
                           element_type=np.float16 if self.is_half else np.float32, 
                           shape=x.shape(), 
                           buffer_ptr=x.data_ptr())

        binding.bind_input(name='prompts', 
                           device_type='cuda', 
                           device_id=0, 
                           element_type=np.int64, 
                           shape=prompts.shape, 
                           buffer_ptr=prompts.data_ptr())
        binding.bind_output('y')
        binding.bind_output('k')        
        binding.bind_output('v')        
        binding.bind_output('y_emb')        
        self.fsdec.run_with_iobinding(binding)

        y = binding.get_outputs()[0]
        k = binding.get_outputs()[1]
        v = binding.get_outputs()[2]
        y_emb = binding.get_outputs()[3]
        return y, k, v, y_emb

    def stage_decode(self, y, k, v, y_emb):
        binding = self.sdec.io_binding()

        binding.bind_input(name='iy', 
                           device_type=y.device_name(), 
                           device_id=0, 
                           element_type=np.int64, 
                           shape=y.shape(), 
                           buffer_ptr=y.data_ptr())

        binding.bind_input(name='ik', 
                           device_type=k.device_name(), 
                           device_id=0, 
                           element_type=np.float16 if self.is_half else np.float32, 
                           shape=k.shape(), 
                           buffer_ptr=k.data_ptr())

        binding.bind_input(name='iv', 
                           device_type=v.device_name(), 
                           device_id=0, 
                           element_type=np.float16 if self.is_half else np.float32, 
                           shape=v.shape(), 
                           buffer_ptr=v.data_ptr())

        binding.bind_input(name='iy_emb', 
                           device_type=y_emb.device_name(), 
                           device_id=0, 
                           element_type=np.float16 if self.is_half else np.float32, 
                           shape=y_emb.shape(), 
                           buffer_ptr=y_emb.data_ptr())

        binding.bind_output('y')
        binding.bind_output('k')        
        binding.bind_output('v')        
        binding.bind_output('y_emb')        
        binding.bind_output('logits')        
        binding.bind_output('samples')        
        self.sdec.run_with_iobinding(binding)

        y = binding.get_outputs()[0]
        k = binding.get_outputs()[1]
        v = binding.get_outputs()[2]
        y_emb = binding.get_outputs()[3]
        logits = binding.get_outputs()[4]
        samples = binding.get_outputs()[5]
        return y, k, v, y_emb, logits, samples

    def infer_panel(
        self,
        all_phoneme_ids,
        prompts,
        bert_feature,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        print('t2s_model_onnx_rt.infer_panel')
        if not self.is_half:
            if bert_feature.dtype == torch.float16:
                bert_feature = bert_feature.type(torch.float32)
        if 1:
            print(all_phoneme_ids.shape, prompts.shape, bert_feature.shape)
        x = self.encode(all_phoneme_ids, bert_feature)
        prefix_len = prompts.shape[1]

        y, k, v, y_emb = self.first_stage_decode(x, prompts)

        times = []
        stop = False
        for idx in range(1, 1500):
            start = time.time()
            enco = self.stage_decode(y, k, v, y_emb)
            times.append(time.time() - start)
            y, k, v, y_emb, logits, samples = enco
            if early_stop_num != -1 and (y.shape()[1] - prefix_len) > early_stop_num:
                stop = True
            logits = torch.from_numpy(logits.numpy())
            samples = torch.from_numpy(samples.numpy())
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                break
        print(times)
        y = torch.from_numpy(y.numpy())
        y = y.to(device=all_phoneme_ids.device)
        y[0, -1] = 0
        return y, idx        

if __name__ == '__main__':
    onnx = T2SModelOnnxRT('onnx/test', is_half=False)

    all_phoneme_ids = np.load('test_data/onnx-x.npy')
    prompts = np.load('test_data/onnx-prompts.npy')
    bert = np.load('test_data/onnx-bert.npy')

    start = time.time()
    y, idx = onnx.infer_panel(torch.from_numpy(all_phoneme_ids).to('cuda'), 
        torch.from_numpy(prompts).to('cuda'),
        torch.from_numpy(bert).to('cuda'))
    print('time cost:', time.time() - start)
    print(idx)
    print(y)
