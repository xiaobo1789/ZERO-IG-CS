import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference  # 导入损失函数相关类
from utils import blur, pair_downsampler  # 导入工具函数：模糊处理、下采样
from torch.utils.checkpoint import checkpoint


class Denoise_1(nn.Module):
    # 第一级去噪模块（轻量级卷积网络）
    def __init__(self, chan_embed=48):
        super(Denoise_1, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # LeakyReLU激活函数（防止死亡神经元）
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)  # 输入3通道（RGB），输出48通道，3x3卷积（保持尺寸）
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)  # 48→48通道，3x3卷积
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)  # 48→3通道，1x1卷积（恢复输入通道数）

    def forward(self, x):
        # 前向传播：输入x→卷积→激活→卷积→激活→卷积→输出
        x = self.act(checkpoint(self.conv1, x))
        x = self.act(checkpoint(self.conv2, x))
        x = checkpoint(self.conv3, x)
        return x


class Denoise_2(nn.Module):
    # 第二级去噪模块（处理6通道输入：3通道图像+3通道光照图）
    def __init__(self, chan_embed=96):
        super(Denoise_2, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(6, chan_embed, 3, padding=1)  # 输入6通道，输出96通道
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)  # 96→96通道
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)  # 96→6通道（恢复输入通道数）

    def forward(self, x):
        # 前向传播：同Denoise_1，处理6通道输入
        x = self.act(checkpoint(self.conv1, x))
        x = self.act(checkpoint(self.conv2, x))
        x = checkpoint(self.conv3, x)
        return x


class Enhancer(nn.Module):
    # 图像增强模块（残差连接的卷积网络）
    def __init__(self, layers, channels):
        super(Enhancer, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation  # 计算填充（保持尺寸）

        # 输入卷积：3→channels通道，ReLU激活
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        # 中间卷积块：channels→channels通道，带BN和ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),  # 批归一化（加速训练，防止过拟合）
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()  # 存储多个卷积块（用于残差连接）
        for i in range(layers):
            self.blocks.append(self.conv)  # 添加layers个卷积块

        # 输出卷积：channels→3通道，Sigmoid激活（输出范围[0,1]）
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # 前向传播：输入→初始卷积→多轮残差卷积→输出卷积
        fea = checkpoint(self.in_conv, input)  # 初始特征提取，使用checkpoint
        for conv in self.blocks:
            fea = fea + checkpoint(conv, fea)  # 残差连接（当前特征+卷积输出），使用checkpoint
        fea = checkpoint(self.out_conv, fea)  # 输出光照图，使用checkpoint
        fea = torch.clamp(fea, 0.0001, 1)  # 限制范围（避免除零错误）
        return fea


class Network(nn.Module):
    # 主网络（训练时使用，包含增强和去噪模块，以及损失计算）
    def __init__(self):
        super(Network, self).__init__()
        self.enhance = Enhancer(layers=3, channels=64)  # 增强模块（3层残差块，64通道）
        self.denoise_1 = Denoise_1(chan_embed=48)  # 第一级去噪模块
        self.denoise_2 = Denoise_2(chan_embed=48)  # 第二级去噪模块
        # 损失函数相关
        self._l2_loss = nn.MSELoss()  # L2损失
        self._l1_loss = nn.L1Loss()   # L1损失
        self._criterion = LossFunction()  # 自定义总损失
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # 平均池化（用于局部特征）
        self.TextureDifference = TextureDifference()  # 纹理差异计算


    def enhance_weights_init(self, m):
        # 增强模块的权重初始化
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)  # 卷积权重：正态分布（0, 0.02）
            if m.bias != None:
                m.bias.data.zero_()  # 偏置初始化为0

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)  # BN权重：正态分布（1, 0.02）

    def denoise_weights_init(self, m):
        # 去噪模块的权重初始化（同增强模块）
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        # 前向传播：输入低光图像→输出多个中间结果（用于损失计算）
        eps = 1e-4  # 防止除零的小值
        input = input + eps  # 输入加小值
        input.requires_grad_(True)  # 确保输入有梯度

        # 下采样输入图像得到两个子图（用于多尺度处理）
        L11, L12 = pair_downsampler(input)
        # 第一级去噪（处理下采样后的子图）
        L_pred1 = L11 - checkpoint(self.denoise_1, L11)
        L_pred2 = L12 - checkpoint(self.denoise_1, L12)
        # 对原始输入去噪
        L2 = input - checkpoint(self.denoise_1, input)
        L2 = torch.clamp(L2, eps, 1)  # 限制范围

        # 增强模块：生成光照图s2，移除detach()以保留梯度
        s2 = checkpoint(self.enhance, L2)  # 关键修改：移除detach()
        s21, s22 = pair_downsampler(s2)  # 下采样光照图
        # 计算增强后的图像（输入/光照图）
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        # 多尺度增强结果（基于下采样的光照图）
        H11 = L11 / s21
        H11 = torch.clamp(H11, eps, 1)
        H12 = L12 / s22
        H12 = torch.clamp(H12, eps, 1)

        # 第二级去噪（处理增强图+光照图的拼接），移除detach()
        concat1 = torch.cat([H11, s21], 1)  # 关键修改：移除detach()
        H3_pred = concat1 - checkpoint(self.denoise_2, concat1)
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]  # 去噪后的增强图（前3通道）
        s13 = H3_pred[:, 3:, :, :]  # 去噪后的光照图（后3通道）

        # 另一路第二级去噪，移除detach()
        concat2 = torch.cat([H12, s22], 1)  # 关键修改：移除detach()
        H4_pred = concat2 - checkpoint(self.denoise_2, concat2)
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        # 对原始增强图H2进行第二级去噪，移除detach()
        concat3 = torch.cat([H2, s2], 1)  # 关键修改：移除detach()
        H5_pred = concat3 - checkpoint(self.denoise_2, concat3)
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]  # 最终去噪结果
        s3 = H5_pred[:, 3:, :, :]  # 最终光照图

        # 计算纹理差异（用于损失约束）
        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff= self.TextureDifference(H3_denoised1, H3_denoised2)

        # 计算模糊版本（用于颜色一致性损失）
        H1 = L2 / s2
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)  # 模糊处理
        H3_blur = blur(H3)

        # 返回所有中间结果（用于损失计算）
        return L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur

    def _loss(self, input):
        # 计算总损失（调用forward得到中间结果，再传入损失函数）
        outputs = self(input)
        loss = self._criterion(input, *outputs)

        # 调用自定义损失函数计算总损失
        return loss


class Finetunemodel(nn.Module):
    # 微调模型（测试时使用，仅保留推理必要的模块）
    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        # 同Network的核心模块
        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)

        # 加载预训练权重
        base_weights = torch.load(weights, map_location='cuda:0')  # 加载权重文件
        pretrained_dict = base_weights
        model_dict = self.state_dict()  # 当前模型的参数字典
        # 过滤出预训练权重中与当前模型匹配的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # 更新模型参数
        self.load_state_dict(model_dict)  # 加载参数

    def weights_init(self, m):
        # 权重初始化（同Network）
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        # 前向传播（仅输出最终增强和去噪结果）
        eps = 1e-4
        input = input + eps
        # 第一级去噪
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)
        # 增强（生成光照图）
        s2 = self.enhance(L2)
        # 计算增强图
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)
        # 第二级去噪（得到最终去噪结果）
        concat = torch.cat([H2, s2], 1)
        H5_pred = concat - self.denoise_2(concat)
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]  # 最终去噪结果
        return H2, H3  # 返回增强图和去噪图