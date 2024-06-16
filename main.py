import argparse
import fnmatch
import os
import sys
from enum import IntEnum

import torch.nn as nn
import torch
from torchvision.transforms import transforms

import P_loader
import coder
import ot


class PathType(IntEnum):
    train = 0
    test = 1
    model = 2
    feature = 3


def path(path_type: PathType, root_dir: str) -> str:
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
    elif path_type == PathType.feature:
        return os.path.join(root_dir, "feature", "feature.pt")


def train_coder(dim: int, coder_dir: str) -> coder.Autoencoder:
    data_path = path(PathType.train, coder_dir)
    test_path = path(PathType.test, coder_dir)
    model_path = path(PathType.model, coder_dir)
    feature_path = path(PathType.feature, coder_dir)

    encoder = coder.Autoencoder(latent_dim=dim).cuda()

    img_transform = transforms.Compose([transforms.ToTensor()])
    dataset = P_loader.P_loader(root=data_path, transform=img_transform)
    testset = P_loader.P_loader(root=test_path, transform=img_transform)
    coder.train(encoder, dataset, testset, model_path)
    coder.refine(encoder, dataset, model_path)

    coder.extract_features(encoder, dataset, feature_path)
    return encoder


def train_ot(feature_dir: str, ot_dir: str):
    ot_model_path = path(PathType.model, ot_dir)
    ot_feature_path = path(PathType.feature, ot_dir)
    ot.compute_ot(feature_dir, ot_model_path, ot_feature_path)


def train_transformer():
    pass


def train(args: argparse.Namespace) -> coder.Autoencoder:
    coder_dir = args.coder_dir
    ot_dir = args.ot_dir

    # train encoder & decoder
    encoder = train_coder(args.latent_dim, coder_dir)
    # train ot: forward & backward, two ot
    train_ot(path(PathType.feature, coder_dir), ot_dir)
    # train transformer
    train_transformer()
    return encoder


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
    parser.add_argument("--train", help="whether to train", dest='train', type=bool, default=True)
    parser.add_argument("--predict", help="whether to predict", dest='predict', type=bool, default=True)
    parser.add_argument("--coder-dir", help='path_type to coder root dir', type=str, metavar="", dest="coder_dir", default="./coder")
    parser.add_argument("--ot-dir", help='path_type to OT root dir', type=str, metavar="", dest="ot_dir", default="./ot")
    parser.add_argument("--latent-dim", help='', type=int, metavar="", dest="latent_dim", default=2)
    args = parser.parse_args()

    if args.train:
        train(args)

    if args.predict:
        input_data = sys.stdin.read()
        lines = input_data.splitlines()

        for line in lines:
            print(predict(args, line))
