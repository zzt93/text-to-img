# import tensorrt as trt
import torch

import config
import util
import vllm_test


def save_model(vocab_size, transformer_model_dir):
    my_config = vllm_test.MyTransformerConfig(vocab_size)
    model = vllm_test.MyTransformerHF(my_config)
    # 加载你训练好的权重
    util.resume_model(model.model, transformer_model_dir, 'Epoch_*_transformer_*.pth')

    # 假设你的模型是 model，输入样例是 example_input
    example_input = torch.tensor(list(bytearray("数字1 (83811) ",  'utf-8')))
    torch.onnx.export(model, example_input, "my_transformer.onnx",
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'},
                                    'output': {0: 'batch_size', 1: 'seq_len'}})


def build_eng():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open("transformer_simplified.onnx", "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_engine(network, config)

    # 保存引擎
    with open("transformer.engine", "wb") as f:
        f.write(engine.serialize())


def run_engine(engine):
    import pycuda.driver as cuda
    import pycuda.autoinit

    # 假设 engine 已经加载
    context = engine.create_execution_context()

    # 分配输入/输出内存
    input_shape = (1, seq_len, input_dim)
    output_shape = (1, seq_len, output_dim)
    d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().nbytes)
    d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().nbytes)

    # 拷贝数据到 GPU
    cuda.memcpy_htod(d_input, input_np)

    # 推理
    context.execute_v2([int(d_input), int(d_output)])

    # 拷贝输出回 CPU
    cuda.memcpy_dtoh(output_np, d_output)