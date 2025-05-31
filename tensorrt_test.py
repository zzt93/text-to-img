import torch
from torch import nn

import config
import minbpe.base
import transformer


class InitialModel(nn.Module):
    def __init__(self, t: transformer.AbsTransformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t

    def forward(self, input_ids, src_key_padding_mask=None, causal_mask=None, k_caches=None, v_caches=None, clone=False):
        # 首次推理逻辑，返回logits和初始KV Cache
        return self.t.forward_full(input_ids, src_key_padding_mask=src_key_padding_mask, causal_mask=causal_mask, k_caches=k_caches, v_caches=v_caches, clone=clone)


class NextModel(nn.Module):
    def __init__(self, t: transformer.AbsTransformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def forward(self, input_ids, src_key_padding_mask=None, causal_mask=None, k_caches=None, v_caches=None, clone=False):
        # 后续推理逻辑，返回logits和更新后的KV Cache
        return self.t.forward_step(input_ids, src_key_padding_mask=src_key_padding_mask, causal_mask=causal_mask, k_caches=k_caches, v_caches=v_caches, clone=clone)


# https://github.com/NVIDIA/TensorRT/blob/HEAD/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20through%20ONNX.ipynb
def save_static_model(my_tokenizer: minbpe.base.Tokenizer, transformer_model):
    transformer_model.eval()
    full_2d_causal_mask = transformer.generate_square_subsequent_mask_bool(config.max_seq_len, device=config.device)
    full_causal_mask = torch.full(size=(1, config.max_seq_len), fill_value=True, device=config.device, dtype=bool)
    k_caches = torch.zeros(transformer_model.num_layers, config.max_seq_len, transformer_model.d_model, device=config.device)
    v_caches = torch.zeros_like(k_caches, device=config.device)

    input = "数字1 (83811)"
    index = len(my_tokenizer.encode(input))
    example_input = torch.tensor(my_tokenizer.encode(input), device=config.device).unsqueeze(0)
    causal_mask = full_2d_causal_mask[:index, :index]

    torch.onnx.export(InitialModel(transformer_model), (example_input, None, causal_mask, k_caches, v_caches), "transformer_full.onnx",
                      input_names=['input_ids', 'causal_mask', 'k_caches', 'v_caches'],
                      output_names=['output', 'new_k_caches', 'new_v_caches'],
                      dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'seq_len'},
                        'causal_mask': {0: 'mask_rows', 1: 'mask_cols'},
                        'output': {0: 'batch_size'}
                      })

    input = " "
    index = index + 1
    example_input = torch.tensor(my_tokenizer.encode(input), device=config.device).unsqueeze(0)
    causal_mask = full_causal_mask[:, :index]
    torch.onnx.export(NextModel(transformer_model), (example_input, None, causal_mask, k_caches, v_caches), "transformer_step.onnx",
                      input_names=['input_ids', 'causal_mask', 'k_caches', 'v_caches'],
                      output_names=['output', 'new_k_caches', 'new_v_caches'],
                      dynamic_axes={
                        'input_ids': {0: 'batch_size'},
                        'causal_mask': {1: 'mask_cols'},
                        'output': {0: 'batch_size'}
                      })


# 模型包含复杂动态控制流（if/for/while 等）
# 使用了 PyTorch 2.x 新特性（如 torch.compile）
# 需要导出支持范围更广的算子
# 传统 torch.onnx.export 无法正确导出的模型
def save_dynamic_model(my_tokenizer: minbpe.base.Tokenizer, transformer_model):
    input = "数字1 (83811) "
    index = len(my_tokenizer.encode(input))
    first = len(input) > 1
    example_input = torch.tensor(my_tokenizer.encode(input), device=config.device).unsqueeze(0)
    full_2d_causal_mask = transformer.generate_square_subsequent_mask_bool(config.max_seq_len, device=config.device)
    causal_mask = full_2d_causal_mask[:index, :index]
    transformer_model.eval()
    k_caches = torch.zeros(transformer_model.num_layers, config.max_seq_len, transformer_model.d_model, device=config.device)
    v_caches = torch.zeros_like(k_caches, device=config.device)

    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)

    kwargs = {"src_key_padding_mask": None, "causal_mask": causal_mask, "k_caches": k_caches, "v_caches": v_caches, "first": torch.tensor(first), "clone": torch.tensor(True)}
    args = (example_input,)
    torch.onnx.dynamo_export(transformer_model, *args, **kwargs
                             , export_options=export_options).save("my_transformer.onnx")
                      # input_names=['input_ids', 'causal_mask', 'k_caches', 'v_caches'],
                      # output_names=['output', 'new_k_caches', 'new_v_caches'],
                      # dynamic_axes={
                      #   'input_ids': {0: 'batch_size', 1: 'seq_len'},
                      #   'causal_mask': {0: 'mask_rows', 1: 'mask_cols'},
                      #   'output': {0: 'batch_size'}
                      # }

