
# if USE_FP16:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt   --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
# else:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt

import torch

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
import torch.nn.functional as F
import time

STEP_TRANSFORMER_TRT = "step_transformer.trt"

FULL_TRANSFORMER_TRT = "full_transformer.trt"


def save_trt(model_path, engine_path, first):
    import tensorrt as trt

    # 初始化 TensorRT 组件
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 解析 ONNX 模型
    parser = trt.OnnxParser(network, logger)
    with open(model_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"ONNX解析错误: {parser.get_error(error)}")
            raise RuntimeError("ONNX 解析失败")

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_name = layer.name

        # 针对问题层设置FP32
        if "softmax" in layer_name.lower() or "norm" in layer_name.lower():
            layer.precision = trt.DataType.FLOAT  # 计算精度设为FP32
            layer.set_output_type(0, trt.DataType.FLOAT)  # 输出类型设为FP32

    # 设置优化配置文件 (动态形状)
    profile = builder.create_optimization_profile()

    # 获取输入名称 (更健壮的方式)
    input_specs = []
    for i in range(network.num_inputs):
        input_name = network.get_input(i).name
        shape = network.get_input(i).shape
        input_specs.append((input_name, shape))
        print(f"输入 {i}: 名称={input_name}, 形状={shape}")

    # 根据您的输入设置动态形状 (假设第一个是input_ids，第二个是causal_mask)
    if len(input_specs) < 2:
        raise ValueError("模型需要至少2个输入")

    input_ids_name = input_specs[0][0]
    causal_mask_name = input_specs[1][0]

    # 设置动态形状范围
    if not first:
        profile.set_shape(
            input=input_ids_name,
            min=(1, 1),
            opt=(1, 1),
            max=(1, 1)
        )
        profile.set_shape(
            input=causal_mask_name,
            min=(1, 1),
            opt=(1, 100),
            max=(1, 500)
        )
        print(f"输入配置: ")
        print(f"  {input_ids_name}: min=[1,1], opt=[1,1], max=[1,1]")
        print(f"  {causal_mask_name}: min=[1,1], opt=[1,100], max=[1,500]")
    else:
        profile.set_shape(
            input=input_ids_name,
            min=(1, 8),
            opt=(1, 16),
            max=(1, 32)
        )
        profile.set_shape(
            input=causal_mask_name,
            min=(8, 8),
            opt=(16, 16),
            max=(32, 32)
        )
    config.add_optimization_profile(profile)

    # 设置内存池限制 (替代旧的max_workspace_size)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # 设置构建标志
    config.set_flag(trt.BuilderFlag.DIRECT_IO)  # 替代 --noDataTransfers
    config.set_flag(trt.BuilderFlag.FP16)  # 开启FP16

    # 构建序列化引擎
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("引擎构建失败")

    # 保存引擎
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"引擎构建成功并保存至: {engine_path}")


class ONNXWrapper():
    def __init__(self, full_file, step_file, target_dtype=np.float32):
        self.target_dtype = target_dtype
        self.first_max_seq_len = 32
        self.max_seq_len = 500
        self.load(full_file, step_file)
        self.full_stream = None
        self.step_stream = None

        self.h_output = np.empty(259, dtype=self.target_dtype)
        self.h_k_cache = np.empty((12,500,768), dtype=self.target_dtype)

        float_byte = np.empty(1, dtype=self.target_dtype).nbytes
        self.d_output = cuda.mem_alloc(1 * 1 * 259 * float_byte)
        self.d_k_caches = cuda.mem_alloc(12 * 500 * 768 * float_byte)
        self.d_v_caches = cuda.mem_alloc(12 * 500 * 768 * float_byte)

    def load(self, full_file, step_file):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        f = open(full_file, "rb")
        self.full_engine = runtime.deserialize_cuda_engine(f.read())
        self.full_context = self.full_engine.create_execution_context()

        f = open(step_file, "rb")
        self.step_engine = runtime.deserialize_cuda_engine(f.read())
        self.step_context = self.step_engine.create_execution_context()

    def allocate_memory(self, batch, full: bool):
        i64_byte = np.empty(1, dtype=np.int64).nbytes
        if full:
            self.d_full_input = cuda.mem_alloc(1 * self.first_max_seq_len * i64_byte)
            self.d_full_attn_mask = cuda.mem_alloc(1 * self.first_max_seq_len * self.first_max_seq_len)

            tensor_names = [self.full_engine.get_tensor_name(i) for i in range(self.full_engine.num_io_tensors)]
            # print(tensor_names)
            assert (len(tensor_names) == 7)

            self.full_context.set_tensor_address(tensor_names[0], int(self.d_full_input))
            self.full_context.set_tensor_address(tensor_names[1], int(self.d_full_attn_mask))
            self.full_context.set_tensor_address(tensor_names[2], int(self.d_k_caches))
            self.full_context.set_tensor_address(tensor_names[3], int(self.d_v_caches))
            self.full_context.set_tensor_address(tensor_names[4], int(self.d_output))
            self.full_context.set_tensor_address(tensor_names[5], int(self.d_k_caches))
            self.full_context.set_tensor_address(tensor_names[6], int(self.d_v_caches))

            self.full_stream = cuda.Stream()
        else:
            self.d_step_input = cuda.mem_alloc(1 * i64_byte)
            self.d_step_attn_mask = cuda.mem_alloc(1 * self.max_seq_len)

            tensor_names = [self.step_engine.get_tensor_name(i) for i in range(self.step_engine.num_io_tensors)]
            # print(tensor_names)
            assert (len(tensor_names) == 7)

            self.step_context.set_tensor_address(tensor_names[0], int(self.d_step_input))
            self.step_context.set_tensor_address(tensor_names[1], int(self.d_step_attn_mask))
            self.step_context.set_tensor_address(tensor_names[2], int(self.d_k_caches))
            self.step_context.set_tensor_address(tensor_names[3], int(self.d_v_caches))
            self.step_context.set_tensor_address(tensor_names[4], int(self.d_output))
            self.step_context.set_tensor_address(tensor_names[5], int(self.d_k_caches))
            self.step_context.set_tensor_address(tensor_names[6], int(self.d_v_caches))

            self.step_stream = cuda.Stream()

    def predict_full(self, batch, casual_mask):  # result gets copied into output
        if self.full_stream is None:
            self.allocate_memory(batch, True)

        # Transfer input data to device
        self.full_context.set_input_shape('input_ids', batch.shape)
        cuda.memcpy_htod_async(self.d_full_input, batch, self.full_stream)
        self.full_context.set_input_shape('causal_mask', casual_mask.shape)
        cuda.memcpy_htod_async(self.d_full_attn_mask, casual_mask, self.full_stream)
        # Execute model
        self.full_context.execute_async_v3(self.full_stream.handle)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.full_stream)
        # Syncronize threads
        self.full_stream.synchronize()

        return self.h_output

    def print_binding(self, engine):
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

        print("\n=== 引擎张量信息 ===")
        for name in tensor_names:
            # 获取张量模式（输入/输出）
            mode = engine.get_tensor_mode(name)
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)

            print(f"Name: {name}, Mode: {mode}, Dtype: {dtype}, Shape: {shape}")

    def predict_step(self, batch, mask):  # result gets copied into output
        if self.step_stream is None:
            self.allocate_memory(batch, False)

        # Transfer input data to device
        self.step_context.set_input_shape('input_ids', batch.shape)
        cuda.memcpy_htod_async(self.d_step_input, batch, self.step_stream)
        self.step_context.set_input_shape('causal_mask', mask.shape)
        cuda.memcpy_htod_async(self.d_step_attn_mask, mask, self.step_stream)

        # Execute model
        self.step_context.execute_async_v3(self.step_stream.handle)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.step_stream)
        # Syncronize threads
        self.step_stream.synchronize()

        return self.h_output


def run_eng():
    def generate_square_subsequent_mask_bool(sz: int, device) -> torch.Tensor:
        if True:
            # F.scaled_dot_product_attention
            # 生成上三角矩阵（不包含对角线），False 表示需要遮蔽的位置
            upper_tri = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
            # 转换成布尔掩码（False 表示需要遮蔽的位置）
            mask = upper_tri == 0
        else:
            # 生成上三角矩阵（不包含对角线），True 表示需要遮蔽的位置
            mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
        return mask


    import numpy as np
    PRECISION = np.float32
    trt_model = ONNXWrapper(FULL_TRANSFORMER_TRT, STEP_TRANSFORMER_TRT, target_dtype=PRECISION)

    input_batch = np.array([[230, 149, 176, 229, 173, 151, 49, 32, 40, 56, 51, 56, 49, 49, 41]], dtype=np.int64)
    index = input_batch.shape[1]
    causal_mask = generate_square_subsequent_mask_bool(index, device='cpu')

    predictions = trt_model.predict_full(input_batch, causal_mask.numpy())
    # print(k_cache)
    probabilities = F.softmax(torch.tensor(predictions), dim=0)
    max_index = torch.argmax(probabilities)
    print(chr(max_index), end="")

    start = time.perf_counter()

    while max_index.item() != 258:
        input_batch = np.array([[max_index]], dtype=np.int64)
        index = index + 1
        causal_mask = torch.full(size=(1, index), fill_value=True, device='cpu', dtype=bool)
        predictions = trt_model.predict_step(input_batch, causal_mask.numpy())
        # print(predictions)
        probabilities = F.softmax(torch.tensor(predictions), dim=0)
        max_index = torch.argmax(probabilities)
        if max_index.item() == 258:
            break
        print(chr(max_index), end="")

    end = time.perf_counter()
    print(f"\nElapsed time: {end - start} seconds")



run_eng()

# tensorrt 10.x运行pytorch导出的onnx格式的，自回归transformer。这个transformer支持kv cache。在使用tensorrt 10.x时，我想先分配比较大的kv cache内存在gpu上，然后每次调用直接修改这块内存，不要重新分配内存。但是导出onnx，不允许in-placement修改kv cache。所以
# 1. 我在pytorch代码中应该使用index_copy 还是index_copy_
# 2. 我在运行的时候，是否还要将kv cache在gpu、cpu之间拷贝
# 3. 是否可以不拷贝，给出代码
