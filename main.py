"""
Usage:
Training:
python main.py --data_path ./data/CelebA/Img \
               --epochs 20 \
               --batch_size 32 \
               --lrG 0.0002 \
               --lrD 0.0001 \
               --image_size 64
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from models.stylegan import Generator, Discriminator
from utils.dataset import get_celeba_dataloader

def visualize_data(dataloader, device, out_dir="outputs"):
    """
    从 dataloader 里取一批图像可视化，确认加载的数据是正常的人脸。
    """
    sample = next(iter(dataloader))
    images, _ = sample
    print("Sample data shape:", images.shape)  # e.g. (batch_size, 3, 64, 64)
    images = images.to(device)

    os.makedirs(out_dir, exist_ok=True)
    grid = make_grid(images, nrow=8, normalize=True, value_range=(-1,1))
    save_image(grid, f"{out_dir}/check_data.png")
    print(f"Saved a batch of real images to {out_dir}/check_data.png for debugging.")

def train_stylegan(dataloader, G, D, criterion, optimizerG, optimizerD, device, nz=100, epochs=30):
    G.train()
    D.train()

    for epoch in range(1, epochs+1):
        for i, (real_imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # 真实标签平滑: 从1.0改为0.9，假标签仍为0
            label_real = torch.full((batch_size,), 0.9, device=device)  # [0.9, 0.9, ...]
            label_fake = torch.zeros(batch_size, device=device)

            # ========== 1) 训练判别器 (D) ========== #
            # a) 用真实图像
            output_real = D(real_imgs)
            loss_real = criterion(output_real, label_real)

            # b) 用假图像
            noise = torch.randn(batch_size, nz, device=device)
            fake_imgs = G(noise).detach()
            output_fake = D(fake_imgs)
            loss_fake = criterion(output_fake, label_fake)

            lossD = loss_real + loss_fake
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()

            # ========== 2) 训练生成器 (G) 多次 ========== #
            # 给生成器多一些更新机会，让它更快追上判别器
            multiple_G_updates = 2  # 可自行调大或调小
            for _ in range(multiple_G_updates):
                noise = torch.randn(batch_size, nz, device=device)
                gen_imgs = G(noise)
                output_gen = D(gen_imgs)
                # 生成器希望判别器输出接近0.9(真实标签)
                lossG = criterion(output_gen, label_real)

                optimizerG.zero_grad()
                lossG.backward()
                optimizerG.step()

            if (i+1) % 100 == 0:
                print(f"Epoch[{epoch}/{epochs}] Batch[{i+1}/{len(dataloader)}] "
                      f"LossD: {lossD.item():.4f}, LossG: {lossG.item():.4f}")

        sample_images(G, epoch, device, nz=nz, out_dir='outputs')

def sample_images(G, epoch, device, nz=100, sample_num=16, out_dir="outputs"):
    G.eval()
    with torch.no_grad():
        noise = torch.randn(sample_num, nz, device=device)
        samples = G(noise).cpu()

    grid = make_grid(samples, nrow=int(sample_num**0.5), normalize=True, value_range=(-1,1))
    os.makedirs(out_dir, exist_ok=True)
    save_image(grid, f"{out_dir}/epoch_{epoch}_samples.png")

    plt.figure(figsize=(5,5))
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.axis('off')
    plt.savefig(f"{out_dir}/epoch_{epoch}_samples_matplot.png")
    plt.close()

    G.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/CelebA/Img',
                        help='Path to CelebA dataset.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lrG', type=float, default=0.0002, help='Generator learning rate.')
    parser.add_argument('--lrD', type=float, default=0.0001, help='Discriminator learning rate (lower).')
    parser.add_argument('--nz', type=int, default=100, help='Dimension of the input noise.')
    parser.add_argument('--image_size', type=int, default=64, help='Try smaller resolution for debugging.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 数据加载 (分辨率调低到 64x64，便于测试)
    dataloader = get_celeba_dataloader(args.data_path, args.image_size, args.batch_size, args.num_workers)

    # 2) 数据可视化调试
    visualize_data(dataloader, device)

    # 3) 初始化模型
    from models.stylegan import Generator, Discriminator
    G = Generator(nz=args.nz, w_dim=512, num_mapping_layers=8, image_channels=3, feature_map_base=256).to(device)
    D = Discriminator(image_channels=3, ndf=32).to(device)

    # 4) 权重初始化
    def weights_init(m):
        classname = m.__class__.__name__
        if 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'Linear' in classname:
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    G.apply(weights_init)
    D.apply(weights_init)

    # 5) 损失函数 & 优化器
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(G.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    optimizerD = optim.Adam(D.parameters(), lr=args.lrD, betas=(0.5, 0.999))

    # 6) 训练
    train_stylegan(dataloader, G, D, criterion, optimizerG, optimizerD,
                   device, nz=args.nz, epochs=args.epochs)

    # 7) 保存模型
    os.makedirs('outputs', exist_ok=True)
    torch.save(G.state_dict(), 'outputs/G_stylegan.pth')
    torch.save(D.state_dict(), 'outputs/D_stylegan.pth')
    print("Training complete! Models saved in 'outputs/' directory.")

if __name__ == '__main__':
    main()