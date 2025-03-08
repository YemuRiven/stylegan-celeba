import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_celeba_dataloader(data_path, image_size=128, batch_size=32, num_workers=2):
    transform = transforms.Compose([
        transforms.CenterCrop(178),    # 原始图像大小为 178x218
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # 归一化到 [-1,1]
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    # 这里假设 data_path 指向上级目录，如 data/img_align_celeba
    # ImageFolder 需要再往里一层，但实际要根据你放置数据的结构自行调整

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return dataloader
