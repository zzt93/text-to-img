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


class Autoencoder(nn.Module):
    def __init__(self, dim_z=100, dim_c=3, dim_f=16):
        super(Autoencoder, self).__init__()
        self.dim_c = dim_c
        self.dim_z = dim_z

        self.block1 = nn.Sequential(
            # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
            nn.Conv2d(3, dim_f, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block2 = nn.Sequential(
            # [-1, 256, 8, 8]
            nn.Conv2d(dim_f, dim_f * 2, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block3 = nn.Sequential(
            # [-1, 512, 4, 4]
            nn.Conv2d(dim_f * 2, dim_f * 4, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 4),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(dim_f * 4, dim_f * 8, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block5 = nn.Sequential(
            # [-1, 1 + cc_dim + dc_dim, 1, 1]
            nn.Conv2d(dim_f * 8, dim_z, 4, 1, 0)
        )

        self.block6 = nn.ConvTranspose2d(dim_z, dim_f * 8, 4, 1, 0)

        self.block7 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 8, dim_f * 4, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 4),
            nn.ReLU(),
        )

        self.block8 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 4, dim_f * 2, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 2),
            nn.ReLU(),
        )

        self.block9 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 2, dim_f, 4, 2, 1),
            nn.BatchNorm2d(dim_f),
            nn.ReLU(),
        )

        self.block10 = nn.Sequential(
            nn.ConvTranspose2d(dim_f, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encoder(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    def decoder(self, z):
        x6 = self.block6(z)
        x7 = self.block7(x6)
        x8 = self.block8(x7)
        x9 = self.block9(x8)
        x10 = self.block10(x9)
        return x10

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x6 = self.block6(x5)
        x7 = self.block7(x6)
        x8 = self.block8(x7)
        x9 = self.block9(x8)
        x10 = self.block10(x9)
        return x10, x5



def refine(model: Autoencoder, dataset: P_loader, model_path: str, batch_size: int=512, num_epochs: int=10, resume: bool = True, learning_rate: float = 2e-5):
    r"""
    Refine the model, just use MSE
    :param model:
    :param dataset:
    :param model_path:
    :param batch_size:
    :param num_epochs:
    :param resume:
    :param learning_rate:
    :return:
    """
    for param in model.block1.parameters():
        param.requires_grad = False
    for param in model.block2.parameters():
        param.requires_grad = False
    for param in model.block3.parameters():
        param.requires_grad = False
    for param in model.block4.parameters():
        param.requires_grad = False
    for param in model.block5.parameters():
        param.requires_grad = False

    # for test_data in testloader:
    #     test_img, _, _ = test_data
    #     break
    if resume:
        for file in os.listdir(model_path):
            if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                model.load_state_dict(torch.load(os.path.join(model_path, file)))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    # save input test image
    # save_image(test_img[:64], os.path.join(img_save_path, 'test_image_input.png'))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
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
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch, num_epochs, loss.item()))
            loss_train += loss.item()
            count_train += 1


        loss_train /= count_train
        loss_test = 0
        # out, _ = model(test_img.cuda())
        # pic = out.data.cpu()
        # save_image(pic[:64], os.path.join(img_save_path,
        #                                   'Epoch_{}_test_image_{:04f}_{:04f}.png'.format(epoch, loss_train,
        #                                                                                  loss_test)))

        torch.save(model.state_dict(), os.path.join(model_path,
                                                    'Epoch_{}_sim_refine_autoencoder_{:04f}_{:04f}.pth'.format(
                                                        epoch, loss_train, loss_test)))


def train(model: Autoencoder, dataset: P_loader, testset: P_loader, model_path: str, loss_weight: float = 1e-5, batch_size: int=512, num_epochs: int=10, resume: bool = True, learning_rate: float = 2e-5):
    r"""
    trains the autoencoder: loss = MSE + loss_weight * torch.norm(z, 1)

    :param model:
    :param dataset:
    :param testset:
    :param model_path:
    :param loss_weight:
    :param batch_size:
    :param num_epochs:
    :param resume:
    :param learning_rate:
    :return:
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

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
            print('epoch [{}/{}], loss1:{:.4f}, loss2:{:.4f}'
                  .format(epoch, num_epochs, loss1.item(), loss2.item()))
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
        # save_image(pic[:64], os.path.join(img_save_path,
        #                                   'Epoch_{}_test_image_{:04f}_{:04f}.png'.format(epoch, loss_train,
        #                                                                                  loss_test)))

        torch.save(model.state_dict(), os.path.join(model_path,
                                                    'Epoch_{}_sim_autoencoder_{:04f}_{:04f}.pth'.format(epoch,
                                                                                                        loss_train,
                                                                                                        loss_test)))


def extract_features(model: Autoencoder, dataset: P_loader, latent_dim: int, feature_save_path: str, batch_size: int=512):
    dataloader_stable = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    features = torch.empty([len(dataset), latent_dim], dtype=torch.float, requires_grad=False, device='cpu')
    i = 0
    for data in dataloader_stable:
        img, _, _ = data
        img = img.cuda()
        img.requires_grad = False
        # ===================forward=====================
        z = model.encoder(img.detach())
        features[i:i + img.shape[0], :] = z.squeeze().detach().cpu()
        i += img.shape[0]
        print('Extracted {}/{} features...'.format(i, len(dataset)))

    features = features[:i]
    torch.save(features, feature_save_path)


def decode_features(model: Autoencoder, gen_im_path: str, model_path: str, gen_feature_path: str, gen_im_pair_path: str, batch_size: int=512):
    for file in os.listdir(model_path):
        if fnmatch.fnmatch(file, 'Epoch_*_sim_refine_autoencoder*.pth'):
            model.load_state_dict(torch.load(os.path.join(model_path, file)))
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
        # save_image(pic_ori, os.path.join(gen_im_pair_path, 'img_{0:03d}_ori.png'.format(i)))
        # y_rec = y[i + num_ids,:,:,:]
        # save_image(y_rec.cpu(), os.path.join(gen_im_pair_path, 'img_{0:03d}_rec.png'.format(i)))
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