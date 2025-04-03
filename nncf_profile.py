import nncf
import torch
from nncf import NNCFConfig
from torch import nn
from torch.utils.data import DataLoader

import config
import transformer
from nncf.torch import create_compressed_model

import minbpe.base
import util
from transformer import PaddingTextDataset
import os

quantization_config = NNCFConfig({
    "compression": {
        "algorithm": "quantization",
        # 预设模式：平衡精度与速度
        "preset": "performance",  # 可选 ["mixed", "performance", "accuracy"]

        # 初始化校准配置
        "initializer": {
            "num_init_samples": 256,  # 校准样本数量 (根据显存调整)
            "range": {
                "type": "mean_min_max",  # 校准方法
                "num_init_samples": 64  # 范围校准样本数
            },
            # 混合精度初始化（适用于Transformer）
            "precision": {
                "type": "hawq",  # 自动混合精度搜索
                "bits": [4, 8]  # 允许的量化位宽
            }
        },

        # 激活值量化参数
        "activations": {
            "mode": "asymmetric",  # 非对称量化更适合注意力输出
            "per_channel": False,
            "clamp_quantile": 0.999,  # 处理激活值离群点
            # 针对GELU激活的特殊处理
            "scale_sharing": {
                "type": "group_wise",
                "groups_count": 4
            }
        },

        # 权重量化参数
        "weights": {
            "mode": "symmetric",
            "per_channel": True,  # 通道级量化提升精度
            "granularity": "perchannel",
            "level_low": -127,  # 8-bit量化范围
            "level_high": 127
        },

        # 针对Transformer结构的特殊配置
        "scope_overrides": {
            # 保护注意力机制层
            "activations": {
                "*.attn": "symmetric",  # 注意力层保持对称量化
                "*.value_proj": "asymmetric"  # 值投影使用非对称
            },
            # 保护残差连接
            "weights": {
                "*.residual.*": {"mode": "asymmetric"}
            }
        },

        # 排除敏感层
        "ignored_scopes": [
            "*embedding*",  # embedding
            "*pos_encoder*",  # 保护位置编码
            "*norm*", # norm层
            "*_attn*"
        ],

        # 高级优化参数
        "advanced": {
            # 溢出处理（适用于大模型）
            "overflow_fix": "first_layer_only",
            # 批归一化适应（如果模型包含BN）
            "batchnorm_adaptation": {
                "num_bn_adaptation_samples": 128
            },
            # 量化误差控制
            "maximal_accuracy_degradation": 1.0  # 允许1%精度损失
        }
    }
})


def get_tokenizer(root_dir):
    return train_tokenizer(root_dir, {})


def train_tokenizer(root_dir: str, train_opt: dict):
    print('train tokenizer [train_opt={}, dir={}]'.format(train_opt, root_dir))
    model = minbpe.regex_impl.RegexTokenizer()
    return transformer.train_tokenizer(model, root_dir, **train_opt)


def get_transformer(transformer_root: str, vocab_size, train_opt, d_model=768, nhead=12, num_layers=12,
                    dim_feedforward=3072, dropout=0.1, **kwargs):
    print('transformer [train_opt={}, transformer_root={}, vocab_size={}]'.format(train_opt, transformer_root,
                                                                                  vocab_size))
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = transformer.MyTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                      dim_feedforward=dim_feedforward, dropout=dropout).to(device)
    if False:
        num_encoder_layers = num_layers
        num_decoder_layers = num_layers
        max_seq_length = 512
        model = CrossAttentionTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, max_seq_length, dropout).to(device)
    return model


def generate_calibration_data(samples: int, transformer_model: transformer.AbsTransformer,
                              tokenizer: minbpe.base.Tokenizer, latent_dim: int) -> None:
    res = []
    for i in range(samples):
        user_input = "数字{}".format(i % 10)
        user_input = util.replace_number(user_input)
        res.append(user_input)
        #res.append(transformer.run_transformer(transformer_model, tokenizer, user_input, force_dim=latent_dim))
    p = config.path(config.PathType.train, "./transformer", config.transformer_calib_data_file)
    print(p, res)
    util.save_data(p, res, "")


# 1. 环境准备
from nncf import Dataset as NNCfDataset
def prepare_calibration_data(batch_size=8, seq_length=512):
    p = config.path(config.PathType.train, "./transformer", config.transformer_calib_data_file)
    texts = util.load_texts(p)
    texts = [
        {
            "input_ids": sample[:-1]
        }
        for sample in texts
    ]

    return NNCfDataset(texts)



# 5. 量化评估流程
def collect_metrics(model, eval_loader, compression_ctrl):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    """自动化指标收集"""
    model.eval()
    model.to(device)

    # 内存占用分析
    memory_stats = torch.cuda.memory_stats(device)

    # 推理速度测试
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    # 精度指标收集
    total_ppl = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch['input_ids'].to(device)

            # 推理时间测量
            starter.record()
            outputs = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

            # 困惑度计算
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1))
            total_ppl += torch.exp(loss).item()

    return {
        "perplexity": total_ppl / len(eval_loader),
        "latency(ms)": sum(timings) / len(timings),
        "memory_usage(MB)": memory_stats['allocated_bytes.all.current'] / 1024 ** 2,
        "quantization_metrics": compression_ctrl.statistics().to_dict()
    }


class TransformerWrapper(nn.Module):
    def __init__(self, core_model, tokenizer):
        super().__init__()
        self.core_model = core_model
        self.tokenizer = tokenizer

    def forward(self, input_ids: str):
        return transformer.run_transformer(
            self.core_model,
            self.tokenizer,
            input_ids,
            force_dim=18
        )
# 6. 执行评估
def eval_use_nncf(tokenizer, my_transformer):
    device = nncf.TargetDevice.GPU if torch.cuda.is_available() else nncf.TargetDevice.CPU
    #calibration_dataset = prepare_calibration_data(tokenizer=tokenizer)
    calibration_dataset = prepare_calibration_data()

    wrapped_model = TransformerWrapper(my_transformer, tokenizer)

    for name, module in wrapped_model.named_modules():
        print(f"Name: {name}, Type: {type(module)}")

    # 转换为NNCFConfig对象
    # nncf_config = NNCFConfig(quantization_config)
    # 4. 创建量化模型并自动收集指标
    compressed_model = nncf.quantize(
        wrapped_model,
        calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
        preset=nncf.QuantizationPreset.MIXED,
        target_device=device,
        ignored_scope=nncf.IgnoredScope([
            # 1. 嵌入层（精确匹配）
            "core_model.embedding",
            "core_model.pos_encoder",

            # 2. 所有LayerNorm及其子层（跨层级匹配）
            "*.*.norm*",  # 匹配 layers.0.norm1 等
            "*.*.ln*",  # 冗余匹配其他可能的命名变体

        ])
    )

    for name, module in compressed_model.named_modules():
        if hasattr(module, 'weight_scale'):
            print(f"{name}:\n Scale={module.weight_scale}\n Zero-point={module.weight_zero_point}")
        if hasattr(module, 'activation_scale'):
            print(f"{name} 激活统计:\n Scale={module.activation_scale}")

    compression_ctrl = compressed_model.nncf.get_compression_controller()
    stats = compression_ctrl.statistics().quantization
    # 5. 收集基础统计信息
    print("Quantization statistics:")
    print(compression_ctrl.statistics().to_str())

    metrics = collect_metrics(compressed_model, calibration_dataset, )

    # 输出关键指标
    print(f"量化后指标：")
    print(f"困惑度(PPL): {metrics['perplexity']:.2f}")
    print(f"单次推理延迟: {metrics['latency(ms)']:.2f}ms")
    print(f"显存占用: {metrics['memory_usage(MB)']:.2f}MB")

    # 输出量化细节
    print("\n量化层统计：")
    for layer, stats in metrics['quantization_metrics']['aq_per_layer'].items():
        print(f"{layer}: \n  量化误差 {stats['mean']:.4f} ± {stats['std_dev']:.4f}")

    # 保存量化模型
    torch.save({
        'model': compressed_model.state_dict(),
        'quant_metadata': compression_ctrl.get_compression_state()
    }, 'quantized_model.pth')


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer_root = './tokenizer'
    transformer_root = './transformer'
    transformer_opt = {}
    tokenizer = get_tokenizer(tokenizer_root)
    print(transformer_root, len(tokenizer.vocab), transformer_opt, **transformer_opt)
    my_transformer = get_transformer(transformer_root, len(tokenizer.vocab), transformer_opt, **transformer_opt)
    print(type(my_transformer))
    transformer_model_dir = config.directory(config.PathType.model, transformer_root)
    util.resume_model(my_transformer, transformer_model_dir, 'Epoch_*_transformer_*.pth')

    my_transformer.eval()
    latent_dim = 18
    test_sample_num = 50

    calib_data_path = config.path(config.PathType.train, "./transformer", config.transformer_calib_data_file)
    if not os.path.exists(calib_data_path):
        print("Generating calibration data...")
        generate_calibration_data(
            samples=256,  # Matches num_init_samples in config
            transformer_model=my_transformer,
            tokenizer=tokenizer,
            latent_dim=latent_dim
        )


    print("Starting quantization process...")
    eval_use_nncf(tokenizer, my_transformer)

    print("Quantization process completed successfully!")

#
if __name__ == "__main__":

    main()
