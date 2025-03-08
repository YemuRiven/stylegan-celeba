# StyleGAN on CelebA

本项目使用 PyTorch 对 [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 数据集进行训练，构建一个简化版的 **StyleGAN** 模型，并生成新的头像图像。

## 特性

- 使用简化版 StyleGAN 架构，包括映射网络 (Mapping Network) 与合成网络 (Synthesis Network)
- 对 CelebA 数据进行预处理、裁剪与归一化
- 训练完成后可从标准正态分布采样，生成多样化的新头像

## 环境安装

1. 克隆本仓库：
    ```bash
    git clone https://github.com/YemuRiven/stylegan-celeba.git
    cd stylegan-celeba
    ```

2. 安装依赖（Conda 或 pip 均可）：
    ```bash
    conda env create -f environment.yml
    conda activate stylegan-celeba
    ```

## 数据准备

1. 从 [CelebA 官方地址](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 下载数据集，放置到 `data/` 文件夹下，结构类似：
 ```
 data/CelebA
    ├── Anno        (标注信息)
    ├── Eval        (验证/测试信息)
    └── Img
       └── img_align_celeba
          ├── 000001.jpg
          ├── 000002.jpg
          ├── ...
 ```
2. 在运行脚本时，将 `--data_path` 参数设置为图像所在目录

## 运行项目

训练 StyleGAN 模型：
```bash
python main.py --data_path ./data/CelebA/Img \
               --epochs 20 \
               --batch_size 32 \
               --lrG 0.0002 \
               --lrD 0.0001 \
               --image_size 64
