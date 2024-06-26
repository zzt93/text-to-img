import fnmatch
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import pdb
from torchvision import transforms
import scipy.io as sio
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import P_loader

_28 = 28

_64 = 64


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=100, dim_color=3, dim_f=16, img_dim=64):
        r"""
        :param latent_dim: 
        :param dim_color: represent the number of color channels (e.g., RGB images where dim_c=3).
        :param dim_f: represent the number of feature maps or filters typically used in convolutional layers.
        """
        super(Autoencoder, self).__init__()
        self.dim_c = dim_color
        self.dim_z = latent_dim
        step = [2, 2, 2, 2, 1]
        padding = [1, 1, 1, 1, 0]
        if img_dim == _64:
            pass
        elif img_dim == _28:
            padding[0] = padding[0] + int((32 - img_dim)/2)
        else:
            raise Exception("unimplemented")

        self.block1 = nn.Sequential(
            # kernel 4*4
            # [batch_size, dim_color=3, img.shape0=32, img.shape1=32] -> [batch_size, dim_f, img.shape0/stride=16, img.shape1/stride=16]
            nn.Conv2d(dim_color, dim_f, 4, step[0], padding[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        block2 = nn.Sequential(
            # ->[batch_size, dim*2, shape0/4=8, shape1/4=]
            nn.Conv2d(dim_f, dim_f * 2, 4, step[1], padding[1]),
            nn.BatchNorm2d(dim_f * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        block3 = nn.Sequential(
            # ->[batch_size, dim*4, shape0/8, shape1/8=]
            nn.Conv2d(dim_f * 2, dim_f * 4, 4, step[2], padding[2]),
            nn.BatchNorm2d(dim_f * 4),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.mid_block = nn.Sequential()
        self.mid_block.append(block2)
        self.mid_block.append(block3)

        last_dim = dim_f * 4
        if img_dim == _64:
            block4 = nn.Sequential(
                # ->[batch_size, dim*8, shape0/16, shape1/16]
                nn.Conv2d(dim_f * 4, dim_f * 8, 4, step[3], padding[3]),
                nn.BatchNorm2d(dim_f * 8),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.mid_block.append(block4)
            last_dim = dim_f * 8


        self.block5 = nn.Sequential(
            # ->[batch_size, latent_dim, shape0/16*1, shape1/16*1]
            # 2维卷积层，其中输入通道数为dim_f * 8，输出通道数为latent_dim，卷积核大小为4，步长为1，填充为0
            nn.Conv2d(last_dim, latent_dim, 4, step[4], padding[4])
        )

        self.block6 = nn.ConvTranspose2d(latent_dim, last_dim, 4, step[4], padding[4])

        self.reverse_mid_block = nn.Sequential()
        if img_dim == _64:
            block7 = nn.Sequential(
                nn.ConvTranspose2d(last_dim, dim_f * 4, 4, step[3], padding[3]),
                nn.BatchNorm2d(dim_f * 4),
                nn.ReLU(),
            )
            self.reverse_mid_block.append(block7)

        block8 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 4, dim_f * 2, 4, step[2], padding[2]),
            nn.BatchNorm2d(dim_f * 2),
            nn.ReLU(),
        )

        block9 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 2, dim_f, 4, step[1], padding[1]),
            nn.BatchNorm2d(dim_f),
            nn.ReLU(),
        )
        self.reverse_mid_block.append(block8)
        self.reverse_mid_block.append(block9)

        self.block10 = nn.Sequential(
            nn.ConvTranspose2d(dim_f, 3, 4, step[0], padding[0]),
            nn.Tanh()
        )

    def encoder(self, x):
        x = self.block1(x)
        for block in self.mid_block:
            x = block(x)
        x = self.block5(x)
        return x

    def decoder(self, z):
        z = self.block6(z)
        for block in self.reverse_mid_block:
            z = block(z)
        z = self.block10(z)
        return z

    def forward(self, x):
        x = self.block1(x)
        for block in self.mid_block:
            x = block(x)
        y = self.block5(x)

        y = self.block6(y)
        for block in self.reverse_mid_block:
            y = block(y)
        z = self.block10(y)
        return z, y

    def opt_eval(self):
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.mid_block.parameters():
            param.requires_grad = False
        for param in self.block5.parameters():
            param.requires_grad = False


def refine(model: Autoencoder, dataloader: DataLoader, model_path: str, num_epochs: int=10, resume: bool = True, learning_rate: float = 2e-5):
    r"""
    Refine the model, just use MSE
    :param model:
    :param model_path:
    :param batch_size:
    :param num_epochs:
    :param resume:
    :param learning_rate:
    :return:
    """

    # for test_data in testloader:
    #     test_img, _, _ = test_data
    #     break
    if resume:
        for file in os.listdir(model_path):
            if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                model.load_state_dict(torch.load(os.path.join(model_path, file)))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # save input test image
    # save_image(test_img[:64], os.path_type.join(img_save_path, 'test_image_input.png'))

    for epoch in range(num_epochs):
        count_train = 0
        loss_train = 0.0
        for data in dataloader:
            img, _, _ = data
            img = Variable(img).cuda()
            # ===================forward=====================
            output, z = model(img)
            loss = criterion(output, img)
            # loss2 = torch.norm(z, 1)
            # loss = loss1 + lmda * loss2
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
            print('coder refine epoch [{}/{}], loss:{:.4f}'.format(epoch, num_epochs, loss.item()))
            loss_train += loss.item()
            count_train += 1


        loss_train /= count_train
        loss_test = 0
        # out, _ = model(test_img.cuda())
        # pic = out.data.cpu()
        # save_image(pic[:64], os.path_type.join(img_save_path,
        #                                   'Epoch_{}_test_image_{:04f}_{:04f}.png'.format(epoch, loss_train,
        #                                                                                  loss_test)))

        torch.save(model.state_dict(), os.path.join(model_path,
                                                    'Epoch_{}_sim_refine_autoencoder_{:04f}_{:04f}.pth'.format(
                                                        epoch, loss_train, loss_test)))


def train(model: Autoencoder, dataloader: DataLoader, testloader: DataLoader, model_path: str, loss_weight: float = 1e-5,
          num_epochs: int=10, resume: bool = True, learning_rate: float = 2e-5):
    r"""
    trains the autoencoder: loss = MSE + loss_weight * torch.norm(z, 1)

    :param model:
    :param model_path:
    :param loss_weight:
    :param num_epochs:
    :param resume:
    :param learning_rate:
    :return:

    Args:
        testloader ():
        dataloader ():
    """
    # for test_data in testloader:
    #     test_img, _, _ = test_data
    #     break

    if resume:
        for file in os.listdir(model_path):
            if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                model.load_state_dict(torch.load(os.path.join(model_path, file)))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        count_train = 0
        loss_train = 0.0
        count_test = 0
        loss_test = 0.0
        for data in dataloader:
            img, _, _ = data
            img = Variable(img).cuda()
            # ===================forward=====================
            output, z = model(img)
            loss1 = criterion(output, img)
            loss2 = torch.norm(z, 1)
            loss = loss1 + loss_weight * loss2
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
            print('coder epoch [{}/{}], loss1:{:.4f}, loss2:{:.4f}'.format(epoch, num_epochs, loss1.item(), loss2.item()))
            loss_train += loss.item()
            count_train += 1

        for data in testloader:
            img, _, _ = data
            img = Variable(img).cuda()
            output, _ = model(img)
            loss = criterion(output, img)
            loss_test += loss.item()
            count_test += 1

        loss_train /= count_train
        loss_test /= count_test

        # out, _ = model(test_img.cuda())
        # pic = out.data.cpu()
        # save_image(pic[:64], os.path_type.join(img_save_path,
        #                                   'Epoch_{}_test_image_{:04f}_{:04f}.png'.format(epoch, loss_train,
        #                                                                                  loss_test)))

        torch.save(model.state_dict(), os.path.join(model_path,
                                                    'Epoch_{}_sim_autoencoder_{:04f}_{:04f}.pth'.format(epoch,
                                                                                                        loss_train,
                                                                                                        loss_test)))


def extract_features(model: Autoencoder, dataset: P_loader, feature_save_path: str, batch_size: int=512):
    model.opt_eval()
    # Create a DataLoader to iterate over the dataset with specified parameters
    dataloader_stable = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    # Pre-allocate a tensor to store features for the entire dataset
    # The shape is [number of samples in the dataset, dimension of the encoded features]
    features = torch.empty([len(dataset), model.dim_z], dtype=torch.float, requires_grad=False, device='cpu')
    # Initialize a counter to keep track of the processed samples
    i = 0
    # Loop over batches of data in the DataLoader
    for data in dataloader_stable:
        # Unpack the data, since only the images (first item) are needed
        # TODO save mapping<path,feature> for train_transformer
        img, _, _ = data
        # Move the images to the GPU
        img = img.cuda()
        # Disable gradient computation for these images to save memory and computations
        img.requires_grad = False
        # detach() will not track gradients
        # Pass the images through the encoder of the model to obtain the encoded features
        z = model.encoder(img.detach())
        # Store the encoded features in the pre-allocated tensor, converting them to CPU
        features[i:i + img.shape[0], :] = z.squeeze().detach().cpu()
        # Update the counter
        i += img.shape[0]
        # Print the progress of the feature extraction
        print('Extracted {}/{} features...'.format(i, len(dataset)))
    # Truncate the features tensor to the actual number of processed samples
    features = features[:i]
    # Save the extracted features to the specified path_type
    torch.save(features, feature_save_path)


def decode_features(model: Autoencoder, gen_im_path: str, gen_feature_path: str, gen_im_pair_path: str, batch_size: int=512):

    feature_dict = sio.loadmat(gen_feature_path)
    features = feature_dict['features']
    ids = feature_dict['ids']

    num_feature = features.shape[0]
    num_ids = num_feature
    z = torch.from_numpy(features).cuda()
    z = z.view(num_feature, -1, 1, 1)
    with torch.no_grad():
        y = model.decoder(z)

    # =====================generate reconstructed-generated image pairs===========
    for i in range(num_ids):
        # pic_ori = dataset[ids[0, i]][0]
        # save_image(pic_ori, os.path_type.join(gen_im_pair_path, 'img_{0:03d}_ori.png'.format(i)))
        # y_rec = y[i + num_ids,:,:,:]
        # save_image(y_rec.cpu(), os.path_type.join(gen_im_pair_path, 'img_{0:03d}_rec.png'.format(i)))
        y_gen = y[i, :, :, :]
        save_image(y_gen.cpu(), os.path.join(gen_im_pair_path, 'img_{0:03d}_gen.png'.format(i)))

        print('Decoding {}/{}...'.format(i, num_ids))

    # =====================generate random images=================================
    y_all = torch.empty([64, 3, 64, 64])
    num_bat_y = features.shape[0] // batch_size
    features = features[:num_bat_y * batch_size, :]
    count = 0
    for i in range(min(num_bat_y, 5)):
        z = torch.from_numpy(features[i * batch_size: (i + 1) * batch_size, :]).cuda()
        z = z.view(batch_size, -1, 1, 1)
        y = model.decoder(z)
        print('Decoding {}/{}...'.format(i * batch_size, features.shape[0]))
        for ii in range(batch_size):
            save_image(y[ii, :, :, :].cpu(), os.path.join(gen_im_path, 'gen_img_{0:03d}.png'.format(count)))
            if count < 64:
                y_all[count] = y[ii, :, :, :].cpu()
            count += 1

    print('Decoding complete. ')