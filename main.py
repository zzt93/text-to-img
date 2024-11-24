import argparse
import ast
import fnmatch
import os
import sys

import config
import minbpe.regex

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import GPT2LMHeadModel

import coder
import ot
import transformer
import util
from config import PathType
from config import path, directory


def train_coder(dim: int, coder_dir: str, option: list, coder_opt: dict) -> coder.AutoEncoder:
    print('train coder [dim={}, dir={}, option={}, train_opt={}]'.format(dim, coder_dir, option, coder_opt))

    model_dir = directory(PathType.model, coder_dir)
    feature_path = path(PathType.result, coder_dir, config.coder_model_file)

    batch_size = 512
    img_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=coder_dir, train=True, transform=img_transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root=coder_dir, train=False, transform=img_transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    img_dime = img_dim(train_loader)
    encoder = coder.LinerAutoEncoder(latent_dim=dim, dim_color=1, img_dim=img_dime, dim_f=32, kernel_size=2).cuda()
    if 'cnn' in option:
        encoder = coder.CnnAutoEncoder(latent_dim=dim, dim_color=1, img_dim=img_dime, dim_f=32, kernel_size=2).cuda()
    elif 'cnn-l' in option:
        encoder = coder.CnnLinearAutoEncoder(latent_dim=dim, dim_color=1, img_dim=img_dime, dim_f=32, kernel_size=2).cuda()

    if 'train' in option:
        coder.train(encoder, train_loader, test_loader, model_dir, **coder_opt)
    if 'refine' in option:
        coder.refine(encoder, train_loader, test_loader, model_dir, **coder_opt)
    if 'extract' in option:
        util.resume_model(encoder, model_dir)
        coder.extract_features(encoder, train_dataset, feature_path, **coder_opt)
    return encoder


def img_dim(loader: DataLoader):
    dataiter = iter(loader)
    images, _ = next(dataiter)
    # torch.Size([batch_size, dim_color, 28, 28])
    return images[0].shape[2]


def train_transformer(dir: str, tokenizer_root: str, train_opt: dict, d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, dropout=0.1, **kwargs):
    print('train transformer [train_opt={}, dir={}]'.format(train_opt, dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer(tokenizer_root)
    vocab_size = len(tokenizer.vocab)
    model = transformer.MyTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout).to(device)

    if False:
        num_encoder_layers = num_layers
        num_decoder_layers = num_layers
        max_seq_length = 512
        model = CrossAttentionTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, max_seq_length, dropout).to(device)

    transformer.train_transformer(model, tokenizer, dir, True, **train_opt)
    transformer.run_transformer(model, dir, "数字1")


def get_tokenizer(root_dir):
    return train_tokenizer(root_dir, {})


def train_tokenizer(root_dir: str, train_opt: dict):
    print('train tokenizer [train_opt={}, dir={}]'.format(train_opt, root_dir))
    model = minbpe.regex.RegexTokenizer()
    return transformer.train_tokenizer(model, root_dir, **train_opt)


def train(args: argparse.Namespace):
    coder_dir = args.coder_dir
    ot_dir = args.ot_dir

    # train encoder & decoder
    if len(args.train_coder) > 2:
        coder_opt = ast.literal_eval(args.coder_opt)
        train_coder(args.latent_dim, coder_dir, args.train_coder.split(','), coder_opt)

    if args.train_ot or args.run_ot:
        ot_model_dir = directory(PathType.model, ot_dir)
        coder_feature_path = path(PathType.result, coder_dir, config.coder_model_file)
        cpu_features = torch.load(coder_feature_path)
        ot_opt = ast.literal_eval(args.ot_opt)
        if args.train_ot:
            ot.compute_ot(cpu_features, ot_model_dir, ot_opt)
        if args.run_ot:
            ot_feature_path = path(PathType.result, ot_dir, config.ot_model_file)
            ot.ot_map(cpu_features, ot_model_dir, ot_feature_path, ot_opt, **ot_opt)

    tokenizer_root = args.tokenizer_dir
    if args.train_tokenizer:
        tokenizer_opt = ast.literal_eval(args.tokenizer_opt)
        train_tokenizer(tokenizer_root, tokenizer_opt)

    if args.train_transformer:
        transformer_root = args.transformer_dir
        transformer_opt = ast.literal_eval(args.transformer_opt)
        train_transformer(transformer_root, tokenizer_root, transformer_opt, **transformer_opt)


def load_model(model_pattern: str, model_path: str, model: nn.Module) -> None:
    if (not os.path.exists(model_path)) or len(os.listdir(model_path)) == 0:
        raise FileNotFoundError("" + model_path)
    for file in os.listdir(model_path):
        if fnmatch.fnmatch(file, model_pattern):
            model.load_state_dict(torch.load(os.path.join(model_path, file)))


def predict(args: argparse.Namespace, user_input: str):

    # use transformer to map text to Gaussian latent token
    # use ot to map Gaussian latent token to real data latent token(filter pattern mixture & )
    # use decoder to map latent token to image
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-coder", help="whether to train coder", dest='train_coder', type=str, default='train,refine,extract')
    parser.add_argument("--latent-dim", help='', type=int, metavar="", dest="latent_dim", default=2)
    parser.add_argument("--coder-option", help="coder train option", dest='coder_opt', type=str, default="{}")
    parser.add_argument("--coder-dir", help='path_type to coder root dir', type=str, metavar="", dest="coder_dir",
                        default="./coder")

    parser.add_argument("--train-ot", help="whether to train ot", dest='train_ot', type=bool, default=False)
    parser.add_argument("--run-ot", help="whether to run ot to generate ot mapping", dest='run_ot', type=bool, default=False)
    parser.add_argument("--ot-option", help="ot option", dest='ot_opt', type=str, default="{}")
    parser.add_argument("--ot-dir", help='path_type to OT root dir', type=str, metavar="", dest="ot_dir",
                        default="./ot")

    parser.add_argument("--train-tokenizer", help="whether to train tokenizer", dest='train_tokenizer', type=bool, default=False)
    parser.add_argument("--tokenizer-option", help="ot option", dest='tokenizer_opt', type=str, default="{}")
    parser.add_argument("--tokenizer-dir", help='path_type to tokenizer root dir', type=str, metavar="", dest="tokenizer_dir",
                        default="./tokenizer")

    parser.add_argument("--train-transformer", help="whether to train transformer", dest='train_transformer', type=bool, default=False)
    parser.add_argument("--transformer-option", help="ot option", dest='transformer_opt', type=str, default="{}")
    parser.add_argument("--transformer-dir", help='path_type to transformer root dir', type=str, metavar="", dest="transformer_dir",
                        default="./transformer")

    parser.add_argument("--predict", help="whether to predict", dest='predict', type=bool, default=True)

    args = parser.parse_args()

    if args.train_coder or args.train_ot or args.run_ot or args.train_tokenizer or args.train_transformer:
        train(args)

    if args.predict:
        print("Enter to generate: ")
        input_data = sys.stdin.read()
        lines = input_data.splitlines()

        for line in lines:
            print(predict(args, line))
