import argparse
import ast
import fnmatch
import os

import config
import minbpe.regex
import minbpe.base

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    encoder = get_encoder(dim, img_dime, option)

    if 'train' in option:
        coder.train(encoder, train_loader, test_loader, model_dir, **coder_opt)
    if 'refine' in option:
        coder.refine(encoder, train_loader, test_loader, model_dir, **coder_opt)
    if 'extract' in option:
        util.resume_model(encoder, model_dir)
        coder.extract_features(encoder, train_dataset, feature_path, **coder_opt)
    return encoder


def get_encoder(dim, img_dime, option: list) -> coder.AutoEncoder:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    encoder = coder.LinerAutoEncoder(latent_dim=dim, dim_color=1, img_dim=img_dime, dim_f=32, kernel_size=2).to(device)
    if 'cnn' in option:
        encoder = coder.CnnAutoEncoder(latent_dim=dim, dim_color=1, img_dim=img_dime, dim_f=32, kernel_size=2).to(device)
    elif 'cnn-l' in option:
        encoder = coder.CnnLinearAutoEncoder(latent_dim=dim, dim_color=1, img_dim=img_dime, dim_f=32,
                                             kernel_size=2).to(device)
    return encoder


def img_dim(loader: DataLoader):
    dataiter = iter(loader)
    images, _ = next(dataiter)
    # torch.Size([batch_size, dim_color, 28, 28])
    return images[0].shape[2]


def train_transformer(transformer_root: str, tokenizer_root: str, train_opt: dict):
    tokenizer = get_tokenizer(tokenizer_root)
    model = get_transformer(transformer_root, len(tokenizer.vocab), train_opt, **train_opt)
    transformer.train_transformer(model, tokenizer, transformer_root, True, **train_opt)


def get_transformer(transformer_root: str, vocab_size, train_opt, d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, dropout=0.1, **kwargs):
    print('transformer [train_opt={}, transformer_root={}, vocab_size={}]'.format(train_opt, transformer_root, vocab_size))

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = transformer.MyTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                      dim_feedforward=dim_feedforward, dropout=dropout).to(device)
    if False:
        num_encoder_layers = num_layers
        num_decoder_layers = num_layers
        max_seq_length = 512
        model = CrossAttentionTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, max_seq_length, dropout).to(device)
    return model


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
            ot.generate_sample_and_ot(cpu_features, ot_model_dir, ot_feature_path, ot_opt, **ot_opt)

    tokenizer_root = args.tokenizer_dir
    if args.train_tokenizer:
        tokenizer_opt = ast.literal_eval(args.tokenizer_opt)
        train_tokenizer(tokenizer_root, tokenizer_opt)

    if args.train_transformer:
        transformer_root = args.transformer_dir
        transformer_opt = ast.literal_eval(args.transformer_opt)
        train_transformer(transformer_root, tokenizer_root, transformer_opt)


def load_model(model_pattern: str, model_path: str, model: nn.Module) -> None:
    if (not os.path.exists(model_path)) or len(os.listdir(model_path)) == 0:
        raise FileNotFoundError("" + model_path)
    for file in os.listdir(model_path):
        if fnmatch.fnmatch(file, model_pattern):
            model.load_state_dict(torch.load(os.path.join(model_path, file)))


def predict(transformer_model: nn.Module, tokenizer: minbpe.base.Tokenizer, ot_raw: ot.OMTRaw, encoder: coder.AutoEncoder, user_input: str, ot_opt: dict):
    # use transformer to map text to Gaussian latent token
    res = transformer.run_transformer(transformer_model, tokenizer, user_input)
    # 数字5(19022) [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
    start, end = res.index('('), res.index(')')
    parsed_list = ast.literal_eval( res[end + 1 + 1:])
    sample_tensor = torch.tensor(parsed_list)

    # use ot to map Gaussian latent token to real data latent token(filter pattern mixture & )
    gen_feature = ot.ot_a_sample(ot_raw, sample=sample_tensor, **ot_opt)
    # use decoder to map latent token to image
    coder.plot_encoder_features(encoder, [gen_feature], [user_input])


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
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        tokenizer_root = args.tokenizer_dir
        transformer_root = args.transformer_dir
        transformer_opt = ast.literal_eval(args.transformer_opt)
        tokenizer = get_tokenizer(tokenizer_root)
        my_transformer = get_transformer(transformer_root, len(tokenizer.vocab), transformer_opt, **transformer_opt)
        transformer_model_dir = config.directory(config.PathType.model, transformer_root)
        util.resume_model(my_transformer, transformer_model_dir, 'Epoch_*_transformer_*.pth')
        my_transformer.eval()

        ot_dir = args.ot_dir
        coder_dir = args.coder_dir
        ot_model_dir = directory(PathType.model, ot_dir)
        coder_feature_path = path(PathType.result, coder_dir, config.coder_model_file)
        cpu_features = torch.load(coder_feature_path, map_location=torch.device(device))
        ot_opt = ast.literal_eval(args.ot_opt)
        points_num = cpu_features.shape[0]
        dim_y = cpu_features.shape[1]
        ot_raw = ot.OMTRaw(cpu_features, points_num, points_num, dim_y, **ot_opt, model_dir=ot_model_dir, count_of_x_in_batch=1)
        ot_raw.set_h(torch.load(ot_raw.h_path(), map_location=torch.device(device)))

        encoder = get_encoder(dim_y, config.mnist_img_dim, ['cnn-l'])
        coder_model_dir = directory(PathType.model, coder_dir)
        util.resume_model(encoder, coder_model_dir)

        print('predict [ot_opt={}]'.format(ot_opt))
        while True:
            user_input = input("Enter a line of text (or type 'exit' to quit): ")

            # Check if the user wants to exit the loop
            if user_input.lower() == 'exit':
                print("Exiting the loop.")
                break

            # Output the input received
            predict(my_transformer, tokenizer, ot_raw, encoder, user_input, ot_opt)
