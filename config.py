import os
from enum import IntEnum

import torch
from transformers import AutoTokenizer


class PathType(IntEnum):
    train = 0
    test = 1
    model = 2
    result = 3


def path(path_type: PathType, root_dir: str, filename: str) -> str:
    dir = directory(path_type, root_dir)
    return os.path.join(dir, filename)


def directory(path_type: PathType, root_dir: str) -> str:
    if path_type == PathType.train:
        dir = os.path.join(root_dir, "train")
        if not os.path.exists(dir):
            raise Exception("no train data dir")
        return dir
    elif path_type == PathType.test:
        dir = os.path.join(root_dir, "test")
        if not os.path.exists(dir):
            raise Exception("no test data dir")
        return dir
    elif path_type == PathType.model:
        dir = os.path.join(root_dir, "model")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    elif path_type == PathType.result:
        dir = os.path.join(root_dir, "result")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir


transformer_train_data_file = "num.text"
transformer_calib_data_file = "calib.text"
tokenizer_model_file = "tokenizer"
coder_model_file = "feature.pt"
ot_model_file = "feature.pt"
mnist_img_dim = 28
max_seq_len = 500
enable_kv_cache = False
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
