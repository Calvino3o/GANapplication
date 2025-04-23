from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
transform = transforms.Compose([
    transforms.ToTensor(), # value of pixel: [0, 255] -> [0, 1]
    transforms.Normalize(mean = (0.5,), std = (0.5,)) # value of tensor: [0, 1] -> [-1, 1]
])
mnist = datasets.MNIST(root='data', train=True, download=True, transform=transform)

batch_size = 100
data_loader = DataLoader(mnist, batch_size, shuffle=True)

class Discriminator(nn.Module):

    def __init__(self, image_size: int, hidden_size: int):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(image_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        out = self.linear2(out)
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        out = self.linear3(out)
        return F.sigmoid(out)

class Generator(nn.Module):
    def __init__(self, image_size: int, latent_size: int, hidden_size: int):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, image_size)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return F.tanh(out)

image_size = 28 * 28
hidden_size = 256
latent_size = 64

G = Generator(image_size=image_size, hidden_size=hidden_size, latent_size=latent_size)
D = Discriminator(image_size=image_size, hidden_size=hidden_size)

#未训练的生成器和判别器输出
# untrained_G_out = G(torch.randn(latent_size))  # Shape: [latent_size]
# untrained_D_out = D(untrained_G_out.view(1, -1))
# print(f"Result from Discriminator: {untrained_D_out.item():.4f}")
# plt.imshow(untrained_G_out.view(28, 28).detach(), cmap='gray')
# plt.show()

num_epochs = 300
device = "cuda:0" if torch.cuda.is_available() else "cpu"
D.to(device=device)
G.to(device=device)

d_optim = optim.Adam(D.parameters(), lr=0.002)
g_optim = optim.Adam(G.parameters(), lr=0.002)

criterion = nn.BCELoss()

d_loss_list, g_loss_list, real_score_list, fake_score_list = ([] for _ in range(4))


#查看 **training.py**```
def run_discriminator_one_batch(d_net: nn.Module,
                                g_net: nn.Module,
                                batch_size: int,
                                latent_size: int,
                                images: torch.Tensor,
                                criterion: nn.Module,
                                optimizer: optim.Optimizer,
                                device: str):
    # 定义真实样本与假样本的标签
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # 使用真实样本训练鉴别器
    outputs = d_net(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # 使用生成样本训练鉴别器
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = g_net(z)
    outputs = d_net(fake_images.detach())
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = d_loss_real + d_loss_fake  # 计算总损失
    d_loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    optimizer.zero_grad()  # 清空梯度

    return d_loss, real_score, fake_score


def run_generator_one_batch(d_net: nn.Module,
                            g_net: nn.Module,
                            batch_size: int,
                            latent_size: int,
                            criterion: nn.Module,
                            optimizer: optim.Optimizer,
                            device: str):
    # 定义生成样本的标签和噪声
    real_labels = torch.ones(batch_size, 1).to(device)
    z = torch.randn(batch_size, latent_size).to(device)

    # 训练生成器
    fake_images = g_net(z)
    outputs = d_net(fake_images)
    g_loss = criterion(outputs, real_labels)  # 计算判别器结果和真实标签的损失
    g_loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    optimizer.zero_grad()  # 清空梯度

    return g_loss, fake_images


def generate_and_save_images(g_net: nn.Module,
                             batch_size: int,
                             latent_size: int,
                             device: str,
                             image_prefix: str,
                             index: int) -> bool:
    def dnorm(x: torch.Tensor):
        min_value = -1
        max_value = 1
        out = (x - min_value) / (max_value - min_value)
        return out.clamp(0, 1)  # plt expects values in [0,1]

    sample_vectors = torch.randn(batch_size, latent_size).to(device)
    fake_images = g_net(sample_vectors)
    fake_images = fake_images.view(batch_size, 1, 28, 28)
    if os.path.exists(image_prefix) is False:
        os.makedirs(image_prefix)
    save_image(dnorm(fake_images), os.path.join(image_prefix, f'fake_images-{index:03d}.png'), nrow=10)
    return True


def run_epoch(d_net: nn.Module,
              g_net: nn.Module,
              train_loader: DataLoader,
              criterion: nn.Module,
              d_optim: optim.Optimizer,
              g_optim: optim.Optimizer,
              batch_size: int,
              latent_size: int,
              device: str,
              d_loss_list: list,
              g_loss_list: list,
              real_score_list: list,
              fake_score_list: list,
              epoch: int, num_epochs: int):
    d_net.train()
    g_net.train()

    for idx, (images, _) in enumerate(train_loader):
        images = images.view(batch_size, -1).to(device)

        # 训练鉴别器
        d_loss, real_score, fake_score = run_discriminator_one_batch(d_net, g_net, batch_size, latent_size, images,
                                                                     criterion, d_optim, device)

        # 训练生成器
        g_loss, _ = run_generator_one_batch(d_net, g_net, batch_size, latent_size, criterion, g_optim, device)
        if (idx + 1) % 300 == 0:
            num = f"Epoch: [{epoch + 1}/{num_epochs}], Batch: [{idx + 1}/{len(train_loader)}]"
            loss_info = f"Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}"
            real_sample_score = f"Real sample score for Discriminator D(x): {real_score.mean().item():.4f}"
            fake_sample_score = f"Fake sample score for Discriminator D(G(x)): {fake_score.mean().item():.4f}"
            print(num + loss_info)
            print(num + real_sample_score)
            print(num + fake_sample_score)

        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item())
        real_score_list.append(real_score.mean().item())
        fake_score_list.append(fake_score.mean().item())


image_prefix = "./sample"

for epoch in range(num_epochs):
    run_epoch(d_net=D, g_net=G,
              train_loader=data_loader, criterion=criterion,
              d_optim=d_optim, g_optim=g_optim,
              batch_size=batch_size, latent_size=latent_size, device=device,
              d_loss_list=d_loss_list, g_loss_list=g_loss_list,
              real_score_list=real_score_list, fake_score_list=fake_score_list,
              epoch=epoch, num_epochs=num_epochs)
    if (epoch+1) % 10 == 0:
        if generate_and_save_images(g_net=G, batch_size=batch_size,
                                 latent_size=latent_size, device=device,
                                 image_prefix=image_prefix, index=epoch+1):

            print(f"Generated images at epoch {epoch+1}")

checkpoint_path = "./checkpoints"

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
torch.save(G.state_dict(), os.path.join(checkpoint_path, "G.pt"))
torch.save(D.state_dict(), os.path.join(checkpoint_path, "D.pt"))


plt.plot(d_loss_list[::200], label="Discriminator Loss")
plt.plot(g_loss_list[::200], label="Generator Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.show()





