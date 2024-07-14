#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import argparse
import os

import numpy as np  # linear algebra
import struct
from array import array
from os.path import join


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    @staticmethod
    def read_images_labels(images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


def download_plot():
    import torch
    from torchvision import datasets, transforms

    # 定义变换
    transform = transforms.Compose([transforms.ToTensor()])

    # 下载并加载测试集
    test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 查看一个批次的数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    print("图像张量形状:", images.shape)
    print("标签:", labels[:10])

    # 查看一张样本图片
    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(np.squeeze(images[0].numpy()), cmap='gray')
    plt.title("标签: {}".format(labels[0].item()))
    plt.show()


if __name__ == '__main__':
    #
    # Verify Reading Dataset via MnistDataloader class
    #
    import random
    import matplotlib.pyplot as plt

    #
    # Set file paths based on added MNIST Datasets
    #
    input_path = './mnist'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')


    #
    # Helper function to show a list of images with their relating titles
    #
    def show_images(images, title_texts):
        cols = 5
        rows = int(len(images) / cols) + 1
        plt.figure(figsize=(30, 20))
        index = 1
        for x in zip(images, title_texts):
            image = x[0]
            title_text = x[1]
            plt.subplot(rows, cols, index)
            plt.imshow(image, cmap=plt.cm.gray)
            if title_text != '':
                plt.title(title_text, fontsize=15)
            index += 1


    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images
    #
    # images_2_show = []
    # titles_2_show = []
    # for i in range(0, 10):
    #     r = random.randint(1, 60000)
    #     images_2_show.append(x_train[r])
    #     titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))
    #
    # for i in range(0, 5):
    #     r = random.randint(1, 10000)
    #     images_2_show.append(x_test[r])
    #     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))
    #
    # show_images(images_2_show, titles_2_show)

    def save_images(dir: str, images: [], labels: []):
        count = {}
        for x in zip(images, labels):
            image = x[0]
            class_label = x[1]
            path = join(dir, str(class_label), str(count.get(class_label, 0)) + ".png")
            count[class_label] = count.get(class_label, 0) + 1
            dir_name = os.path.dirname(path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            plt.imsave(path, image)


    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", help='path_type to train root dir', type=str, metavar="", dest="train_dir", default="./train")
    parser.add_argument("--test-dir", help='path_type to test root dir', type=str, metavar="", dest="test_dir", default="./test")
    args = parser.parse_args()

    save_images(args.train_dir, x_train, y_train)
    print('generate train data done')
    save_images(args.test_dir, x_test, y_test)
    print('generate test data done')
