import argparse
import fnmatch
import os
import sys
from enum import IntEnum

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import coder
import ot


class PathType(IntEnum):
    train = 0
    test = 1
    model = 2
    result = 3


def path(path_type: PathType, root_dir: str, filename: str = "feature.pt") -> str:
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
        return os.path.join(root_dir, "result", filename)


def train_coder(dim: int, coder_dir: str, train_option: list) -> coder.Autoencoder:
    print('train coder [dim={}, dir={}, train_option={}]'.format(dim, coder_dir, train_option))

    model_path = path(PathType.model, coder_dir)
    feature_path = path(PathType.result, coder_dir)

    batch_size = 512
    img_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=coder_dir, train=True, transform=img_transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root=coder_dir, train=False, transform=img_transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    encoder = coder.LinerAutoEncoder(latent_dim=dim, img_dim=img_dim(train_loader)).cuda()
    if 'train' in train_option:
        coder.train(encoder, train_loader, test_loader, model_path)
    if 'refine' in train_option:
        coder.refine(encoder, train_loader, test_loader, model_path)
    if 'extract' in train_option:
        coder.resume_model(encoder, model_path)
        coder.extract_features(encoder, train_dataset, feature_path)
    return encoder


def img_dim(loader: DataLoader):
    dataiter = iter(loader)
    images, _ = next(dataiter)
    # torch.Size([batch_size, dim_color, 28, 28])
    return images[0].shape[2]


def train_transformer():
    pass


def train(args: argparse.Namespace):
    coder_dir = args.coder_dir
    ot_dir = args.ot_dir

    # train encoder & decoder
    if args.train_coder:
        train_coder(args.latent_dim, coder_dir, args.train_coder.split(','))

    ot_model_path = path(PathType.model, ot_dir)
    coder_feature_path = path(PathType.result, coder_dir)
    cpu_features = torch.load(coder_feature_path)
    if args.train_ot:
        ot.compute_ot(cpu_features, ot_model_path)
    else:
        pass
        # ot_feature_path = path(PathType.result, ot_dir)
        # ot.ot_map(cpu_features, ot_model_path, ot_feature_path)

    if args.train_transformer:
        train_transformer()


def load_model(model_pattern: str, model_path: str, model: nn.Module) -> None:
    if (not os.path.exists(model_path)) or len(os.listdir(model_path)) == 0:
        raise FileNotFoundError("" + model_path)
    for file in os.listdir(model_path):
        if fnmatch.fnmatch(file, model_pattern):
            model.load_state_dict(torch.load(os.path.join(model_path, file)))


def predict(args: argparse.Namespace, user_input: str):
    model = coder.Autoencoder(args.latent_dim).cuda()

    # use transformer to map text to Gaussian latent token
    # use ot to map Gaussian latent token to real data latent token(filter pattern mixture & )
    # use decoder to map latent token to image
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-coder", help="whether to train coder", dest='train_coder', type=str, default='train,refine,extract')
    parser.add_argument("--train-ot", help="whether to train ot", dest='train_ot', type=bool, default=False)
    parser.add_argument("--train-transformer", help="whether to train transformer", dest='train_transformer', type=bool, default=False)
    parser.add_argument("--predict", help="whether to predict", dest='predict', type=bool, default=True)
    parser.add_argument("--coder-dir", help='path_type to coder root dir', type=str, metavar="", dest="coder_dir",
                        default="./coder")
    parser.add_argument("--ot-dir", help='path_type to OT root dir', type=str, metavar="", dest="ot_dir",
                        default="./ot")
    parser.add_argument("--latent-dim", help='', type=int, metavar="", dest="latent_dim", default=2)
    args = parser.parse_args()

    if args.train_coder or args.train_ot or args.train_transformer:
        train(args)

    if args.predict:
        input_data = sys.stdin.read()
        lines = input_data.splitlines()

        for line in lines:
            print(predict(args, line))
