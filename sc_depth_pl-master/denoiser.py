import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from imageio import imwrite
from visualization import visualize_depth
from sklearn.metrics import mean_squared_error
from torchvision.transforms import Resize
import cv2

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 定义编码器结构
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 定义中间层结构
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 定义解码器结构
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(370, 1224), mode='bilinear', align_corners=True)  # 添加上采样层
        )

    def forward(self, x):
        # 实现前向传播
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# 自定义数据集类
class DepthDataset(Dataset):
    def __init__(self, pseudo_depth_dir, ground_truth_dir):
        super(DepthDataset, self).__init__()
        self.pseudo_depth_files = sorted(os.listdir(pseudo_depth_dir))
        self.ground_truth_files = sorted(os.listdir(ground_truth_dir))
        self.pseudo_depth_dir = pseudo_depth_dir
        self.ground_truth_dir = ground_truth_dir

    def __len__(self):
        return len(self.pseudo_depth_files)

    def __getitem__(self, idx):
        pseudo_depth_file = self.pseudo_depth_files[idx]
        ground_truth_file = self.ground_truth_files[idx]
        pseudo_depth = np.load(os.path.join(self.pseudo_depth_dir, pseudo_depth_file))
        ground_truth = np.load(os.path.join(self.ground_truth_dir, ground_truth_file))

        # 设置调整后的深度图尺寸
        fixed_size = (1224, 370)

        # 调整深度图尺寸
        pseudo_depth_resized = cv2.resize(pseudo_depth, fixed_size, interpolation=cv2.INTER_AREA)
        ground_truth_resized = cv2.resize(ground_truth, fixed_size, interpolation=cv2.INTER_AREA)

        # 增加通道维度
        pseudo_depth_resized = np.expand_dims(pseudo_depth_resized, axis=0)
        ground_truth_resized = np.expand_dims(ground_truth_resized, axis=2).transpose((2, 0, 1))  # 转置使得形状正确

        return torch.from_numpy(pseudo_depth_resized).float(), torch.from_numpy(ground_truth_resized).float()


def train_unet_model(pseudo_depth_dir, ground_truth_dir):
    # 初始化模型、优化器和损失函数
    model = UNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    # 加载数据
    dataset = DepthDataset(pseudo_depth_dir, ground_truth_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 训练模型
    epochs = 100
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for pseudo_depth, ground_truth in tqdm(dataloader):
            pseudo_depth = pseudo_depth.cuda()
            ground_truth = ground_truth.cuda()

            optimizer.zero_grad()

            output = model(pseudo_depth)
            loss = criterion(output, ground_truth)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}')

    # 保存模型
    torch.save(model.state_dict(), 'unet_checkpoint.pth')


def denoise_depth(input_depth_dir, output_depth_dir, ground_truth_dir, denoiser):
    input_files = sorted(os.listdir(input_depth_dir))

    # 设置调整后的深度图尺寸
    fixed_size = (1224, 370)

    if not os.path.exists(output_depth_dir):
        os.makedirs(output_depth_dir)

    vis_output_dir = os.path.join(output_depth_dir, "vis")
    if not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)

    total_rmse = 0
    num_files = len(input_files)

    for depth_file in tqdm(input_files):
        input_depth_path = os.path.join(input_depth_dir, depth_file)
        depth_data = np.load(input_depth_path)

        # 调整深度图尺寸
        depth_data_resized = cv2.resize(depth_data, fixed_size, interpolation=cv2.INTER_AREA)

        depth_tensor = torch.from_numpy(depth_data_resized).float().unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            denoised_depth = denoiser(depth_tensor).squeeze(0).squeeze(0).cpu().numpy()

        output_depth_path = os.path.join(output_depth_dir, depth_file)
        np.save(output_depth_path, denoised_depth)

        # 计算 RMSE
        ground_truth_path = os.path.join(ground_truth_dir, depth_file)
        ground_truth_depth = np.load(ground_truth_path)

        # 调整深度图尺寸
        ground_truth_depth_resized = cv2.resize(ground_truth_depth, fixed_size, interpolation=cv2.INTER_AREA)

        rmse = np.sqrt(mean_squared_error(ground_truth_depth_resized, denoised_depth))
        total_rmse += rmse

        # 保存可视化深度图
        vis_output_path = os.path.join(vis_output_dir, os.path.splitext(depth_file)[0] + '.jpg')
        vis_depth = visualize_depth(torch.tensor(denoised_depth)).permute(1, 2, 0).numpy() * 255
        imwrite(vis_output_path, vis_depth.astype(np.uint8))

    avg_rmse = total_rmse / num_files
    print("Average RMSE:", avg_rmse)

def main():
    # 训练U-Net模型
    pseudo_depth_dir = 'results/kitti/model_v1/depth'
    ground_truth_dir = 'F:/kitti/kitti/testing/depth'
    # train_unet_model(pseudo_depth_dir, ground_truth_dir)

    # 加载训练好的深度去噪模型
    denoiser = UNet()
    denoiser.load_state_dict(torch.load('unet_checkpoint.pth'))
    denoiser.cuda()
    denoiser.eval()

    # 测试深度去噪效果
    input_depth_dir = 'results/kitti/model_v1/depth'
    output_depth_dir = 'results/kitti/model_v1/denoise_depth'
    denoise_depth(input_depth_dir, output_depth_dir, ground_truth_dir, denoiser)


if __name__ == '__main__':
    main()