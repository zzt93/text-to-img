from transformers import PreTrainedModel, PretrainedConfig

import transformer
import util


class MyTransformerConfig(PretrainedConfig):
    model_type = "gpt2"
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(self, vocab_size=259, d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout


class MyTransformerHF(PreTrainedModel):
    _supports_attention_backend = True
    config_class = MyTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = transformer.MyTransformer(vocab_size=config.vocab_size, d_model=config.d_model, nhead=config.nhead, num_layers=config.num_layers,
                                      dim_feedforward=config.dim_feedforward, dropout=config.dropout)

    def forward(self, input_ids, src_key_padding_mask=None, causal_mask=None, index=None, **kwargs):
        return self.model(input_ids, src_key_padding_mask=src_key_padding_mask, causal_mask=causal_mask, index=index, **kwargs)


def save_model(vocab_size, transformer_model_dir):
    config = MyTransformerConfig(vocab_size)
    model = MyTransformerHF(config)
    # 加载你训练好的权重
    util.resume_model(model.model, transformer_model_dir, 'Epoch_*_transformer_*.pth')

    from transformers import AutoConfig, AutoModel

    AutoConfig.register("gpt2", MyTransformerConfig)
    AutoModel.register(MyTransformerConfig, MyTransformerHF)

    model.save_pretrained("./vllm")
    config.save_pretrained("./vllm")


def run_opt_llm():
    from vllm import LLM, SamplingParams

    # 加载模型（CPU 模式，device="cpu"）
    llm = LLM(model="./vllm", dtype="auto", device="cpu")

    # 设置推理参数
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

    # 推理
    outputs = llm.generate(["数字1 (83811) "], sampling_params=params)
    for output in outputs:
        print(output.outputs[0].text)