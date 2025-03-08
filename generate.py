import argparse
import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# 在 models/stylegan.py 中定义了 Generator 类
# 如果用的是 DCGAN、VAE 等，根据实际情况导入生成器类
from models.stylegan import Generator

def main():
    parser = argparse.ArgumentParser(description="Generate new images using a trained generator.")
    parser.add_argument('--model_path', type=str, default='outputs/G_stylegan.pth',
                        help='Path to the trained generator model (.pth file).')
    parser.add_argument('--out_dir', type=str, default='generated_outputs',
                        help='Directory to save the generated images.')
    parser.add_argument('--num_images', type=int, default=16,
                        help='Number of images to generate.')
    parser.add_argument('--nz', type=int, default=100,
                        help='Dimension of the input noise/vector.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (e.g., "cuda" or "cpu").')
    args = parser.parse_args()

    # --------------------------
    # 1) 创建输出文件夹
    # --------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # --------------------------
    # 2) 定义并加载生成器
    #    请确保以下参数与训练时相同
    # --------------------------
    # 示例：StyleGAN 的 Generator 参数
    # 如果用的是 DCGAN 或其他模型，用对应结构
    G = Generator(
        nz=args.nz,          # 输入噪声维度
        w_dim=512,           # 映射空间维度(仅示例)
        num_mapping_layers=8, # 映射网络层数(仅示例)
        image_channels=3,    # 输出图像通道数
        feature_map_base=256  # 特征图基数(仅示例)
    ).to(args.device)

    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=args.device)
    G.load_state_dict(checkpoint)
    G.eval()

    # --------------------------
    # 3) 采样随机噪声并生成图像
    # --------------------------
    with torch.no_grad():
        # 从标准正态分布采样
        noise = torch.randn(args.num_images, args.nz, device=args.device)
        # 生成图像
        images = G(noise).cpu()  # shape: (num_images, 3, H, W)

    # --------------------------
    # 4) 拼成网格并保存
    # --------------------------
    # 归一化到 [-1,1] 时，可用 (normalize=True, value_range=(-1,1))
    grid = make_grid(images, nrow=int(args.num_images**0.5),
                     normalize=True, value_range=(-1, 1))

    # 保存到文件
    save_image(grid, os.path.join(args.out_dir, 'generated_samples.png'))
    print(f"Generated images saved to {os.path.join(args.out_dir, 'generated_samples.png')}")

    # --------------------------
    # 5) 可选：用 matplotlib 可视化
    # --------------------------
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.axis('off')
    plt.savefig(os.path.join(args.out_dir, 'generated_samples_matplot.png'))
    plt.close()


if __name__ == '__main__':
    main()
