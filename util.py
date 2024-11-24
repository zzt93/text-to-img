import fnmatch
import os
from enum import IntEnum
from glob import glob
import numpy as np
import torch

import matplotlib.pyplot as plt
from torch import nn as nn


def save_labels():
    from torchvision import datasets, transforms

    img_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./coder', train=True, transform=img_transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)

    labels = []
    for data in train_loader:
        _, label = data
        labels += label.tolist()
    torch.save(labels, './coder/result/label.pth')


def get_creation_time(file_path):
    stat = os.stat(file_path)
    try:
        return stat.st_birthtime
    except AttributeError:
        return stat.st_mtime  # Use last modified time as a fallback


def sort_files_by_creation_time(directory, reverse=True):
    files = glob(os.path.join(directory, '*'))
    sorted_files = sorted(files, key=get_creation_time, reverse=reverse)
    return sorted_files


colors = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Pink', 'Brown', 'Black', 'skyblue']


def show_feature_distribution(features, labels: list):
    if features.shape[1] == 3:
        x = features[:, 0]
        y = features[:, 1]
        z = features[:, 2]
        c = labels[0:len(x)]
        c = [colors[k] for k in c]

        # 创建3D图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(x, y, z, c=c, alpha=0.5)
        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(y, x, z, c=c, alpha=0.5)
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(z, y, x, c=c, alpha=0.5)
        ax = fig.add_subplot(224, projection='3d')
        ax.scatter(y, z, x, c=c, alpha=0.5)
        plt.show()
    elif features.shape[1] == 2:
        array = features.numpy()

        m = {}
        for f, l in zip(array, labels):
            f = f.reshape(-1,2)
            if l not in m:
                m[l] = np.array(f)
            else:
                m[l] = np.concatenate((m[l], f), axis=0)

        i = 0
        for l, f in m.items():
            # 拆分为x和y坐标
            x = f[:, 0]
            y = f[:, 1]
            # 画点图
            plt.scatter(x, y, label=l, color=colors[i])
            i += 1

        # 添加标题和坐标轴标签
        plt.title("2D Tensor features")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        # 显示图形
        plt.show()


def removesuffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


# Load texts and labels from files
def load_data(text_file, label_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f]

    return texts, labels


def save_data(filename: str, texts: list, labels):
    with open(filename, 'w', encoding='utf-8') as file:
        for line in texts:
            file.write(line + '\n')
    print(f"Text has been successfully written to {filename}")


def resume_model(model: nn.Module, model_dir: str, file_pattern: str='Epoch_*_sim_autoencoder*.pth'):
    m = file_loss(model_dir, file_pattern)

    for file, loss in m.items():
        print('resume model using {}'.format(file))
        model.load_state_dict(torch.load(file))
        return
    print('no model, start from scratch')


def file_loss(model_dir: str, file_pattern: str):
    m = {}
    for file in glob(os.path.join(model_dir, '*')):
        filename = os.path.basename(file)
        if fnmatch.fnmatch(filename, file_pattern):
            split = removesuffix(filename, ".pth").split("_")
            test_loss = float(split[-1])
            if is_float_string(split[-2]):
                train_loss = float(split[-2])
            else:
                train_loss = 0
            m[file] = train_loss + test_loss
    return dict(sorted(m.items(), key=lambda item: item[1]))


def is_float_string(value):
    if '.' not in value:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

def load_texts(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return texts


def save_model(model, model_dir: str, name: str, loss_pattern: str='Epoch_*_sim_autoencoder*.pth'):
    torch.save(model.state_dict(), os.path.join(model_dir, name))
    m = file_loss(model_dir, loss_pattern)
    i = 0
    for file, loss in m.items():
        i += 1
        if i > 10:
            os.remove(file)
