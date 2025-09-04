import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(), # probability therefore 0 to 1 range
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1), # Weak gradients expected
            nn.Linear(256, img_dim),
            nn.Tanh(), # Normalise data from -1 to 1
        )
    
    def forward(self, z):
        return self.gen(z)


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # Apparently best learning rate for Adam
z_dim = 64
image_dim = 1*28*28
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # (channel - mean) / std
])

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True) # download if not already in folder
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
cost = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake") # generator images
writer_real = SummaryWriter(f"runs/GAN_MNIST/real") # real images
step = 0


for epoch in range(num_epochs):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.view(-1, image_dim).to(device)
        batch_size = real.shape[0]

        # Discriminator training (maximise cost is same as minimising costD_real)
        # Minimise -(E[log(D(x))] + E[log(1-D(G(z)))])
        # Could have k loop for training 
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise) # G(z)
        disc_real = disc(real).view(-1)
        costD_real = cost(disc_real, torch.ones_like(disc_real)) # First Expectation
        disc_fake = disc(fake).view(-1)
        costD_fake = cost(disc_fake, torch.zeros_like(disc_fake)) # Second Expectation
        costD = (costD_real + costD_fake)/2
        # Set gradients to 0 THEN add calculated gradients THEN optimizer takes step
        disc.zero_grad()
        costD.backward(retain_graph=True) # Need graph for generator calculation
        opt_disc.step()

        # Generator training (minimise cost <-> minimising E[log(1-D(G(z)))] <-> maximising log(D(G(z))))
        # Could have k loops for generator training before 1 discriminator training loop
        output = disc(fake).view(-1)
        costG = cost(output, torch.ones_like(output))
        
        gen.zero_grad()
        costG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {costD:.4f}, loss G: {costG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1