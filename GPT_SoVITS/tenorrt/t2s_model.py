import os
import numpy as np
import torch
import time
import ctypes
import tensorrt as trt
from cuda import cuda, cudart
from typing import Optional, List, Union, Tuple

debug = True

TRT_ENGINE_DIR = 'trt/'
TRT_LOGGER = trt.Logger()

def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype, with_host=True):
        nbytes = size * dtype.itemsize

        if with_host:
            host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        else:
            self._host = None

        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        assert self._host is not None
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    def reshape_host(self, shape):
        size = trt.volume(shape)
        self.shaped_host = self.host[:size].reshape(shape)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        if self._host is not None:
            cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def calc_shapes(engine: trt.ICudaEngine, context: trt.IExecutionContext, profile_idx: int):
    ret = {}
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        binding = engine.get_tensor_name(i)
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            shape = engine.get_tensor_profile_shape(binding, profile_idx)[-1]
            context.set_input_shape(binding, shape)
            input_names.append(binding)
        else:
            shape = context.get_tensor_shape(binding)
            output_names.append(binding)

        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        ret[binding] = shape
    print(ret)
    return ret, input_names, output_names

def get_buf_size(engine: trt.ICudaEngine, binding: str, shape: Tuple):
    size = trt.volume(shape)
    trt_type = engine.get_tensor_dtype(binding)

    # Allocate host and device buffers
    try:
        dtype = np.dtype(trt.nptype(trt_type))
        return size * dtype.itemsize
    except TypeError: # no numpy support: create a byte array instead (BF16, FP8, INT4)
        return int(size * trt_type.itemsize)

def alloc_buffer(engine: trt.ICudaEngine, binding: str, shape: Tuple, with_host=True):
    size = trt.volume(shape)
    trt_type = engine.get_tensor_dtype(binding)

    # Allocate host and device buffers
    try:
        dtype = np.dtype(trt.nptype(trt_type))
        bindingMemory = HostDeviceMem(size, dtype, with_host)
    except TypeError: # no numpy support: create a byte array instead (BF16, FP8, INT4)
        size = int(size * trt_type.itemsize)
        bindingMemory = HostDeviceMem(size, with_host)
    return bindingMemory

def get_trt_file(onnx_file):
    name = os.path.basename(onnx_file)
    m, _ = os.path.splitext(name)
    return os.path.join(TRT_ENGINE_DIR, '%s.trt' % m)

def dump_io_tensors(engine: trt.ICudaEngine, profile_idx):
    if not debug:
        return
    ctx = engine.create_execution_context()
    for i in range(engine.num_io_tensors):
        binding = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(binding)
        pshape = engine.get_tensor_profile_shape(binding, profile_idx)        
        dtype = engine.get_tensor_dtype(binding)
        mode = engine.get_tensor_mode(binding)        
        if mode == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(binding, pshape[-1])
        else:
            shape = ctx.get_tensor_shape(binding)
        print(binding, shape, pshape, dtype, mode)

def build_engine(onnx_file, trt_file, set_profile, memory_pool_limit):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        0
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, memory_pool_limit
        )  # 256MiB
        # Parse model file
        if not os.path.exists(onnx_file):
            print(
                "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                    onnx_file
                )
            )
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file))
        with open(onnx_file, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("Completed parsing of ONNX file")
        print(
            "Building an engine from file {}; this may take a while...".format(
                onnx_file
            )
        )
        profile = builder.create_optimization_profile()
        set_profile(profile)
        config.add_optimization_profile(profile)            

        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(trt_file, "wb") as f:
            f.write(plan)
        return engine

class XEncoder:
    def __init__(self, onnx_file, trt_file=None):
        self.onnx_file = onnx_file
        self.trt_file = trt_file
        if self.trt_file is None:
            self.trt_file = get_trt_file(onnx_file)
        self.engine = self.load_or_build_engine()
        self.context = self.engine.create_execution_context()
        self.shapes, self.input_names, self.output_names = calc_shapes(self.engine, self.context, 0)
        self.inputs = {}
        dump_io_tensors(self.engine, 0)

    def load_or_build_engine(self):
        if os.path.exists(self.trt_file):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(self.trt_file))
            with open(self.trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return self.build_engine()

    def build_engine(self):
        def set_profile(profile):
            profile.set_shape("all_phoneme_ids", (1, 1), (1, 50), (1, 512)) 
            profile.set_shape("bert", (1, 1024, 1), (1, 1024, 50), (1, 1024, 512)) 
        return build_engine(self.onnx_file, self.trt_file, set_profile, 1 << 28)

    def alloc_buffers(self, stream):
        self.stream = stream
        self.context.set_optimization_profile_async(0, stream)
        print(self.shapes)
        for binding in self.input_names:
            shape = self.shapes[binding]
            print(binding, shape)
            mem = alloc_buffer(self.engine, binding, shape, True)
            self.inputs[binding] = mem

    def free_buffers(self):
        if not self.inputs:
            return
        for _, m in self.inputs.items():
            m.free()

    def forward(self, all_phoneme_ids, bert, x_output):
        self.context.set_input_shape('all_phoneme_ids', all_phoneme_ids.shape)
        self.context.set_input_shape('bert', bert.shape)

        self.inputs['all_phoneme_ids'].host = all_phoneme_ids
        self.inputs['bert'].host = bert

        for binding, mem in self.inputs.items():
            self.context.set_tensor_address(binding, int(mem.device))
        self.context.set_tensor_address('x', int(x_output.device))

        kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        for binding, inp in self.inputs.items():
            shape = self.context.get_tensor_shape(binding)
            print(binding, shape)
            cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, get_buf_size(self.engine, binding, shape), kind, self.stream))
        # Run inference.
        self.context.execute_async_v3(stream_handle=self.stream)
        # Synchronize the stream
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return self.context.get_tensor_shape('x')

class FirstStageDecoder:
    def __init__(self, onnx_file, trt_file=None):
        self.onnx_file = onnx_file
        self.trt_file = trt_file
        if self.trt_file is None:
            self.trt_file = get_trt_file(onnx_file)
        self.engine = self.load_or_build_engine()
        self.context = self.engine.create_execution_context()
        self.shapes, self.input_names, self.output_names = calc_shapes(self.engine, self.context, 0)
        dump_io_tensors(self.engine, 0)

    def load_or_build_engine(self):
        if os.path.exists(self.trt_file):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(self.trt_file))
            with open(self.trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return self.build_engine()

    def build_engine(self):
        def set_profile(profile):
            profile.set_shape("prompts", (1, 1), (1, 50), (1, 512)) 
            profile.set_shape("x", (1, 1, 512), (1, 50, 512), (1, 512, 512)) 
        return build_engine(self.onnx_file, self.trt_file, set_profile, 1 << 30)

    def alloc_buffers(self, stream):
        self.stream = stream
        self.context.set_optimization_profile_async(0, stream)
        self.prompts = alloc_buffer(self.engine, 'prompts', self.shapes['prompts'], True)
        self.x = alloc_buffer(self.engine, 'x', self.shapes['x'], False)

    def free_buffers(self):
        self.prompts.free()
        self.x.free()

    def forward(self, x, x_shape, prompts, y_out, k_out, v_out, y_emb_out):
        self.context.set_input_shape('x', x_shape)
        self.context.set_input_shape('prompts', prompts.shape)

        self.prompts.host = prompts

        input_names = ['prompts', 'x']
        inputs = [self.prompts, self.x]
        output_names = ['y', 'k', 'v', 'y_emb']
        outputs = [y_out, k_out, v_out, y_emb_out]

        for i, binding in enumerate(input_names+output_names):
            mem = (inputs+outputs)[i]
            self.context.set_tensor_address(binding, int(mem.device))

        kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        cuda_call(cudart.cudaMemcpyAsync(self.prompts.device, self.prompts.host, get_buf_size(self.engine, 'prompts', prompts.shape), kind, self.stream))
        # Run inference.
        self.context.execute_async_v3(stream_handle=self.stream)
        # Synchronize the stream
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return [self.context.get_tensor_shape(binding) for binding in output_names]

class StageDecoder:
    def __init__(self, onnx_file, trt_file=None):
        self.onnx_file = onnx_file
        self.trt_file = trt_file
        if self.trt_file is None:
            self.trt_file = get_trt_file(onnx_file)
        self.engine = self.load_or_build_engine()
        self.context = self.engine.create_execution_context()
        self.shapes, self.input_names, self.output_names = calc_shapes(self.engine, self.context, 0)
        dump_io_tensors(self.engine, 0)

    def load_or_build_engine(self):
        if os.path.exists(self.trt_file):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(self.trt_file))
            with open(self.trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return self.build_engine()

    def build_engine(self):
        def set_profile(profile):
            profile.set_shape("iy", (1, 1), (1, 50), (1, 512)) 
            profile.set_shape("ik", (24, 1, 1, 512), (24, 1, 50, 512), (24, 1, 512, 512)) 
            profile.set_shape("iv", (24, 1, 1, 512), (24, 1, 50, 512), (24, 1, 512, 512)) 
            profile.set_shape("iy_emb", (1, 1, 512), (1, 50, 512), (1, 512, 512)) 
        return build_engine(self.onnx_file, self.trt_file, set_profile, 1 << 32)

    def alloc_buffers(self, stream):
        self.stream = stream
        self.context.set_optimization_profile_async(0, stream)
        self.iy = alloc_buffer(self.engine, 'iy', self.shapes['y'], True)
        self.ik = alloc_buffer(self.engine, 'ik', self.shapes['k'], False)
        self.iv = alloc_buffer(self.engine, 'iv', self.shapes['v'], False)
        self.iy_emb = alloc_buffer(self.engine, 'iy_emb', self.shapes['y_emb'], False)
        self.y = alloc_buffer(self.engine, 'y', self.shapes['y'], True)
        self.k = alloc_buffer(self.engine, 'k', self.shapes['k'], False)
        self.v = alloc_buffer(self.engine, 'v', self.shapes['v'], False)
        self.y_emb = alloc_buffer(self.engine, 'y_emb', self.shapes['y_emb'], False)
        self.logits = alloc_buffer(self.engine, 'logits', self.shapes['logits'], True)
        self.samples = alloc_buffer(self.engine, 'samples', self.shapes['samples'], True)
        self.bindings = [self.iy, self.ik, self.iv, self.iy_emb, 
            self.y, self.k, self.v, self.y_emb, self.logits, self.samples]
        self.input_groups = [
            [self.iy, self.ik, self.iv, self.iy_emb],
            [self.y, self.k, self.v, self.y_emb]
        ]

    def free_buffers(self):
        for m in self.bindings:
            m.free()

    def forward(self, iy, ik, iv, iy_emb, 
        input_shapes,
        y_out, k_out, v_out, y_emb_out, logits_out, samples_out):
        self.context.set_input_shape('iy', input_shapes[0])
        self.context.set_input_shape('ik', input_shapes[1])
        self.context.set_input_shape('iv', input_shapes[2])
        self.context.set_input_shape('iy_emb', input_shapes[3])

        input_names = ['iy', 'ik', 'iv', 'iy_emb']
        inputs = [iy, ik, iv, iy_emb]
        output_names = ['y', 'k', 'v', 'y_emb', 'logits', 'samples']
        outputs = [y_out, k_out, v_out, y_emb_out, logits_out, samples_out]
        output_shapes = [self.context.get_tensor_shape(binding) for binding in output_names]

        for i, binding in enumerate(input_names+output_names):
            mem = (inputs+outputs)[i]
            self.context.set_tensor_address(binding, int(mem.device))

        # Run inference.
        self.context.execute_async_v3(stream_handle=self.stream)
        # Synchronize the stream
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost

        for i, binding in enumerate(output_names):
            if i < 4:
                continue
            out = outputs[i]
            shape = output_shapes[i]
            nbytes = get_buf_size(self.engine, binding, shape)
            cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, nbytes, kind, self.stream))
            out.reshape_host(shape)
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return output_shapes

class T2SModel:
    def __init__(self, onnx_root, is_half=True, prefix=None):
        self.onnx_root = onnx_root
        self.prefix = prefix
        self.is_half = is_half
        self.EOS = 1024
        self.init_sessions()

    def init_sessions(self):        
        prefix = self.prefix
        if prefix is None:
            prefix = os.path.basename(self.onnx_root)
        
        self.encoder = XEncoder(os.path.join(self.onnx_root, '%s_t2s_encoder.onnx' % prefix))
        self.fsd = FirstStageDecoder(os.path.join(self.onnx_root, '%s_t2s_fsdec.onnx' % prefix))
        self.sd = StageDecoder(os.path.join(self.onnx_root, '%s_t2s_sdec.onnx' % prefix))

        self.stream = cuda_call(cudart.cudaStreamCreate())
        self.alloc_buffers()

    def alloc_buffers(self):
        self.encoder.alloc_buffers(self.stream)
        self.fsd.alloc_buffers(self.stream)
        self.sd.alloc_buffers(self.stream)

    def destroy(self):
        self.encoder.free_buffers()
        self.fsd.free_buffers()
        self.sd.free_buffers()
        cuda_call(cudart.cudaStreamDestroy(self.stream))

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
        # print('t2s_model_tensorrt.infer_panel')
        if not self.is_half:
            if bert_feature.dtype == torch.float16:
                bert_feature = bert_feature.type(torch.float32)
        if 1:
            print(all_phoneme_ids.shape, prompts.shape, bert_feature.shape)
        target_device = all_phoneme_ids.device
        all_phoneme_ids = all_phoneme_ids.cpu().numpy()
        prompts = prompts.cpu().numpy()
        bert_feature = bert_feature.cpu().numpy()

        x_shape = self.encoder.forward(all_phoneme_ids, bert_feature, self.fsd.x)
        prefix_len = prompts.shape[1]
        
        y, k, v, y_emb = self.sd.input_groups[0]

        y_shape, k_shape, v_shape, y_emb_shape = self.fsd.forward(x, x_shape, prompts, y, k, v, y_emb)

        times = []
        stop = False
        for idx in range(1, 1500):
            start = time.time()
            shapes = [y_shape, k_shape, v_shape, y_emb_shape]
            iy, ik, iv, iy_emb = self.sd.input_groups[(idx-1)%2]
            y, k, v, y_emb = self.sd.input_groups[idx%2]
            shapes = self.sd.forward(iy, ik, iv, iy_emb, shapes,
                y, k, v, y_emb, self.sd.logits, self.sd.samples)
            times.append(time.time() - start)
            y_shape, k_shape, v_shape, y_emb_shape = shapes[:4]
            if early_stop_num != -1 and (y_shape()[1] - prefix_len) > early_stop_num:
                stop = True
            
            logits, samples = self.sd.logits, self.sd.samples

            if np.argmax(logits.shaped_host, axis=-1)[0] == self.EOS or \
                samples.shaped_host[0, 0] == self.EOS:
                stop = True
            if stop:
                break
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        nbytes = get_buf_size(self.sd.engine, 'y', y_shape)
        cuda_call(cudart.cudaMemcpyAsync(y.host, y.device, nbytes, kind, self.stream))
        y.reshape_host(y_shape)
        cuda_call(cudart.cudaStreamSynchronize(self.stream))

        y = torch.from_numpy(y.shaped_host)
        y = y.to(device=target_device)
        y[0, -1] = 0
        return y, idx

if __name__ == '__main__':
    # encoder = XEncoder('onnx/test/test_t2s_encoder.onnx')
    # fsd = FirstStageDecoder('onnx/test/test_t2s_fsdec.onnx')
    # sd = StageDecoder('onnx/test/test_t2s_sdec.onnx')

    model = T2SModel('onnx/test')

    x = np.load('test_data/onnx-x.npy')
    prompts = np.load('test_data/onnx-prompts.npy')
    bert = np.load('test_data/onnx-bert.npy')

    start = time.time()
    y, idx = model.infer_panel(torch.from_numpy(x), 
        torch.from_numpy(prompts),
        torch.from_numpy(bert))
    print('time cost:', time.time() - start)
    print(idx)
    print(y)
