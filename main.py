import argparse
import fnmatch
import os
import sys
import torch.nn as nn
import torch
from torchvision.transforms import transforms

import P_loader
import coder


def train_coder(dim: int, coder_dir: str) -> None:
    data_path = coder_dir + "/train"
    test_path = coder_dir + "/test"
    model_path = coder_dir + "/model"

    encoder = coder.Autoencoder(dim)

    img_transform = transforms.Compose([transforms.ToTensor(),])
    dataset = P_loader.P_loader(root=data_path, transform=img_transform)
    testset = P_loader.P_loader(root=test_path, transform=img_transform)
    coder.train(encoder, dataset, testset, model_path)
    coder.refine(encoder, dataset, model_path)
    pass


def train_ot():
    pass


def train_transformer():
    pass


def train(args: argparse.Namespace) -> None:
    coder_dir = args.coder_dir

    # train encoder & decoder
    train_coder(2, coder_dir)
    # train ot: forward & backward, two ot
    train_ot()
    # train transformer
    train_transformer()
    pass


def load_model(model_pattern: str, model_path: str, model: nn.Module) -> None:
    if (not os.path.exists(model_path)) or len(os.listdir(model_path)) == 0:
        raise FileNotFoundError("" + model_path)
    for file in os.listdir(model_path):
        if fnmatch.fnmatch(file, model_pattern):
            model.load_state_dict(torch.load(os.path.join(model_path, file)))


def predict(user_input: str):

    # use transformer to map text to Gaussian latent token
    # use ot to map Gaussian latent token to real data latent token(filter pattern mixture & )
    # use decoder to map latent token to image
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="whether to train", dest='train', type=bool, default=True)
    parser.add_argument("--coder-dir", help='path to coder root dir', type=str, metavar="", dest="coder_dir")
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        input_data = sys.stdin.read()
        lines = input_data.splitlines()

        for line in lines:
            print(predict(line))
