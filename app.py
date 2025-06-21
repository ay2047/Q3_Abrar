import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
from PIL import Image

# Same generator architecture as in train.py
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=1, feature_map_gen=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_gen * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_gen * 8, feature_map_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_gen * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_gen * 4, feature_map_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_gen * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_gen * 2, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def load_generator(model_path="generator.pth", device='cpu'):
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    return netG

def generate_images(generator, digit, num_samples=5, latent_dim=100, device='cpu'):
    """
    For demonstration, we'll simply generate random images
    but won't condition on 'digit'. Conditioning would need a 
    conditional GAN architecture or a classifier-based approach.
    """
    noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        generated = generator(noise).cpu()
    # Transform from [-1,1] to [0,255]
    generated = (generated + 1) / 2
    return generated

def main():
    st.title("MNIST Digit Generator")
    st.write("Select a digit (0-9) to generate images.")

    digit = st.selectbox("Choose a digit", list(range(10)), index=0)
    if st.button("Generate Images"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = load_generator(device=device)
        generated = generate_images(generator, digit, num_samples=5, device=device)
        
        st.write(f"Generated images for digit: {digit}")
        for img_tensor in generated:
            img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
            img_pil = Image.fromarray(img_np.astype(np.uint8).squeeze(), mode='L')
            st.image(img_pil, width=128)

if __name__ == "__main__":
    main()