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

def generate_calibration_data(samples: int, transformer_model: nn.Module, tokenizer: minbpe.base.Tokenizer, latent_dim: int) -> None:
    res = []
    for i in range(samples):
        user_input = "数字{}".format(i % 10)
        user_input = util.replace_number(user_input)
        res.append(transformer.run_transformer(transformer_model, tokenizer, user_input, force_dim=latent_dim))
    p = config.path(config.PathType.train, "./transformer", config.transformer_calib_data_file)
    util.save_data(p, res, "")

if
