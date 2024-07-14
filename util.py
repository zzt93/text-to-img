import os
from glob import glob
import numpy as np
import torch

import matplotlib.pyplot as plt


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