import torch

import config
import minbpe.base
import transformer

# https://github.com/NVIDIA/TensorRT/blob/HEAD/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20through%20ONNX.ipynb

def save_static_model(my_tokenizer: minbpe.base.Tokenizer, transformer_model):
    input = "数字1 (83811) "
    index = len(my_tokenizer.encode(input))
    example_input = torch.tensor(my_tokenizer.encode(input), device=config.device).unsqueeze(0)
    full_2d_causal_mask = transformer.generate_square_subsequent_mask_bool(config.max_seq_len, device=config.device)
    causal_mask = full_2d_causal_mask[:index, :index]
    # example_input = torch.tensor(list(bytearray("数字1 (83811) ",  'utf-8')), device=config.device).squeeze()
    transformer_model.eval()
    k_caches = torch.zeros(transformer_model.num_layers, config.max_seq_len, transformer_model.d_model, device=config.device)
    v_caches = torch.zeros_like(k_caches, device=config.device)
    torch.onnx.export(transformer_model, (example_input, None, causal_mask, k_caches, v_caches), "my_transformer.onnx",
                      input_names=['input_ids', 'causal_mask', 'k_caches', 'v_caches'],
                      output_names=['output', 'new_k_caches', 'new_v_caches'],
                      dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'seq_len'},
                        'causal_mask': {0: 'mask_rows', 1: 'mask_cols'},
                        'output': {0: 'batch_size'}
                      })


def save_model(my_tokenizer: minbpe.base.Tokenizer, transformer_model):
    input = " "
    index = len(my_tokenizer.encode(input))
    example_input = torch.tensor(my_tokenizer.encode(input), device=config.device).unsqueeze(0)
    full_2d_causal_mask = transformer.generate_square_subsequent_mask_bool(config.max_seq_len, device=config.device)
    causal_mask = full_2d_causal_mask[:index, :index]
    # example_input = torch.tensor(list(bytearray("数字1 (83811) ",  'utf-8')), device=config.device).squeeze()
    transformer_model.eval()
    k_caches = torch.zeros(transformer_model.num_layers, config.max_seq_len, transformer_model.d_model, device=config.device)
    v_caches = torch.zeros_like(k_caches, device=config.device)

    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)

    kwargs = {"src_key_padding_mask": None, "causal_mask": causal_mask, "k_caches": k_caches, "v_caches": v_caches}
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

