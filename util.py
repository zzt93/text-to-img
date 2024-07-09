import coder
import matplotlib.pyplot as plt
import numpy as np  # linear algebra


def show_encoder_features(features):
    encoder = coder.Autoencoder(latent_dim=2, img_dim=28)
    for i in features:
        i = i.view(1, -1, 1, 1)
        y = encoder.decoder(i)
        print(y)
        np_array = y.view(3, 28, 28).detach().numpy()
        # For convert the tensor shape from (channels, height, width) to (height, width, channels)
        np_array = np.transpose(np_array, (1, 2, 0))
        plt.imshow(np_array, cmap=plt.cm.gray)
        plt.show()


def img_paths():
    from torchvision.transforms import transforms
    import P_loader
    from torch.utils.data import DataLoader

    img_transform = transforms.Compose([transforms.ToTensor()])
    dataset = P_loader.P_loader(root='./coder/train', transform=img_transform)
    dataloader_stable = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=True, num_workers=4)
    p = []
    for data in dataloader_stable:
        _, _, paths = data
        p += paths

    print(p)
