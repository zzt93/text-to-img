
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
    def __init__(self, file, target_dtype=np.float32):
        self.target_dtype = target_dtype
        self.max_seq_len = 500
        self.load(file)
        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.max_seq_len, dtype=self.target_dtype)  # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * self.output.nbytes)
        self.d_attn_mask = cuda.mem_alloc(1 * self.max_seq_len * self.max_seq_len)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        print(tensor_names)
        assert (len(tensor_names) == 3)

        self.context.set_tensor_address(tensor_names[0], int(self.d_input))
        self.context.set_tensor_address(tensor_names[1], int(self.d_attn_mask))
        self.context.set_tensor_address(tensor_names[2], int(self.d_output))

        self.stream = cuda.Stream()

    def predict(self, batch, mask):  # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        self.context.set_binding_shape(0, batch.shape(1))
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        self.context.set_binding_shape(1, (batch.shape(1), batch.shape(1)))
        cuda.memcpy_htod_async(self.d_attn_mask, mask, self.stream)
        # Execute model
        self.context.execute_async_v3(self.stream.handle)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return self.output


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

    full_2d_causal_mask = generate_square_subsequent_mask_bool(500, device='cpu')

    import numpy as np
    PRECISION = np.float32
    trt_model = ONNXWrapper("transformer.trt", target_dtype=PRECISION)

    input_batch = np.array([[230, 149, 176, 229, 173, 151, 49, 32, 40, 56, 51, 56, 49, 49, 41, 32]], dtype=PRECISION)
    index = input_batch.shape[1]
    causal_mask = full_2d_causal_mask[:index, :index]

    predictions = trt_model.predict(input_batch, causal_mask.numpy())
    print(predictions)


run_eng()