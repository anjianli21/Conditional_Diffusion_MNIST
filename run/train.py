from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from datetime import datetime

from model.diffusion_1d import *


class CustomDataset(Dataset):
    def __init__(self, x, c):
        self.x = x
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]

def train_mnist():
    # hardcoding these here
    n_epoch = 20
    batch_size = 256
    n_T = 400  # 500
    device = "cuda:0"
    context_dim = 5
    n_feat = 128  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

    # Create folder
    now = datetime.now()
    formatted_date_time = now.strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'./data/diffusion_outputs10/{formatted_date_time}'
    if not os.path.isfile(save_dir):
        os.mkdir(save_dir)
        print(f"log saved in {save_dir}")

    ddpm = DDPM1D(nn_model=ContextUnet1D(in_channels=3, n_feat=n_feat, context_dim=context_dim), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    # # MNIST dataset
    # tf = transforms.Compose([transforms.ToTensor()])  # mnist is already normalised 0 to 1
    # dataset = MNIST("./data", train=True, download=True, transform=tf)
    # dataset = torch.rand(64, 3, 20)

    # Our own 1d dataset
    # Create random data for x and c
    x = torch.rand(60000, 3, 20)  # 60000 samples, 3 channels, sequence length 20
    c = torch.rand(60000, 5)  # 60000 samples, dimension 5
    dataset = CustomDataset(x, c)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # evaluation of the model
        ddpm.eval()
        with torch.no_grad():
            n_sample = 1

            for w_i, w in enumerate(ws_test):
                x_gen = ddpm.sample(n_sample, (3, 20), device, guide_w=w)

        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train_mnist()

