import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Simple DCGAN-like generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=1, feature_map_gen=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_map_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_gen * 8),
            nn.ReLU(True),
            # state size: (feature_map_gen*8) x 4 x 4
            nn.ConvTranspose2d(feature_map_gen * 8, feature_map_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_gen * 4),
            nn.ReLU(True),
            # state size: (feature_map_gen*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_gen * 4, feature_map_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_gen * 2),
            nn.ReLU(True),
            # state size: (feature_map_gen*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_gen * 2, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (num_channels) x 32 x 32
        )

    def forward(self, x):
        return self.main(x)

# Simple DCGAN-like discriminator
class Discriminator(nn.Module):
    def __init__(self, num_channels=1, feature_map_disc=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, feature_map_disc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_disc, feature_map_disc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_disc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_disc * 2, feature_map_disc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_disc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_disc * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

def train_gan(epochs=1, batch_size=128, latent_dim=100, device='cuda'):
    transform_data = transforms.Compose([
        transforms.Resize(32),  # Adjusting the input size to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            b_size = imgs.size(0)

            # Train Discriminator
            netD.zero_grad()
            real_labels = torch.ones(b_size, device=device)
            fake_labels = torch.zeros(b_size, device=device)
            output_real = netD(imgs)
            loss_real = criterion(output_real, real_labels)

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            output_fake = netD(fake_imgs.detach())
            loss_fake = criterion(output_fake, fake_labels)

            D_loss = loss_real + loss_fake
            D_loss.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output_fake_for_G = netD(fake_imgs)
            G_loss = criterion(output_fake_for_G, real_labels)
            G_loss.backward()
            optimizerG.step()

        print(f"Epoch [{epoch+1}/{epochs}] - D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}")

    # Save the generator for later use
    torch.save(netG.state_dict(), "generator.pth")
    print("Training complete. Generator saved as generator.pth.")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_gan(epochs=5, device=device)