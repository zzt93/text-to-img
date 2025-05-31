
# if USE_FP16:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt   --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
# else:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt

import torch

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit


class ONNXWrapper():
    def __init__(self, full_file, step_file, target_dtype=np.float32):
        self.target_dtype = target_dtype
        self.first_max_seq_len = 32
        self.max_seq_len = 500
        self.load(full_file, step_file)
        self.full_stream = None
        self.step_stream = None

    def load(self, full_file, step_file):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        f = open(full_file, "rb")
        self.full_engine = runtime.deserialize_cuda_engine(f.read())
        self.full_context = self.full_engine.create_execution_context()

        f = open(step_file, "rb")
        self.step_engine = runtime.deserialize_cuda_engine(f.read())
        self.step_context = self.step_engine.create_execution_context()

    def allocate_memory(self, batch, full: bool):
        type_byte = np.empty(1, dtype=self.target_dtype).nbytes  # Need to set both input and output precisions to FP16 to fully enable FP16
        self.h_output = np.empty(259, dtype=self.target_dtype)
        self.d_output = cuda.mem_alloc(1 * 1 * 259 * type_byte)
        self.d_k_caches = cuda.mem_alloc(12 * 500 * 768 * type_byte)
        self.d_v_caches = cuda.mem_alloc(12 * 500 * 768 * type_byte)
        if full:
            self.d_full_input = cuda.mem_alloc(1 * self.first_max_seq_len * type_byte)
            self.d_full_attn_mask = cuda.mem_alloc(1 * self.first_max_seq_len * self.first_max_seq_len)

            tensor_names = [self.full_engine.get_tensor_name(i) for i in range(self.full_engine.num_io_tensors)]
            print(tensor_names)
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
            self.d_step_input = cuda.mem_alloc(1 * type_byte)
            self.d_step_attn_mask = cuda.mem_alloc(1 * self.max_seq_len)

            tensor_names = [self.step_engine.get_tensor_name(i) for i in range(self.step_engine.num_io_tensors)]
            print(tensor_names)
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
    trt_model = ONNXWrapper("full_transformer.trt", "step_transformer.trt", target_dtype=PRECISION)

    input_batch = np.array([[230, 149, 176, 229, 173, 151, 49, 32, 40, 56, 51, 56, 49, 49, 41]], dtype=PRECISION)
    index = input_batch.shape[1]
    causal_mask = generate_square_subsequent_mask_bool(index, device='cpu')

    predictions = trt_model.predict_full(input_batch, causal_mask.numpy())
    print(predictions)

    input_batch = np.array([[32]], dtype=PRECISION)
    index = index + 1
    causal_mask = torch.full(size=(1, index), fill_value=True, device='cpu', dtype=bool)
    predictions = trt_model.predict_step(input_batch, causal_mask.numpy())
    print(predictions)


run_eng()