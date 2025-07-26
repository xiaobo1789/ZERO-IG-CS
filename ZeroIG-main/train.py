import os
import sys
import time
import glob
import numpy as np
import utils  # 导入工具函数
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn  # 用于优化CUDA卷积
from torch.autograd import Variable
from model import *  # 导入模型相关类
from multi_read_data import DataLoader  # 导入自定义数据加载器
import gc  # 用于垃圾回收

# 限制PyTorch的内存分配
import torch

torch.cuda.set_per_process_memory_fraction(0.9)  # 限制为GPU内存的90%
torch.cuda.empty_cache()

# 解析命令行参数
parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
parser.add_argument('--cuda', default=True, type=bool, help='是否使用CUDA训练')
parser.add_argument('--gpu', type=str, default='0', help='GPU设备ID（多个用逗号分隔）')
parser.add_argument('--seed', type=int, default=2, help='随机种子（保证可复现性）')
parser.add_argument('--epochs', type=int, default=800, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.0003, help='学习率')
parser.add_argument('--save', type=str, default='./EXP/', help='实验结果保存根目录')
parser.add_argument('--model_pretrain', type=str, default='', help='预训练模型路径（可选）')

args = parser.parse_args()  # 解析参数

# 设置CUDA可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 创建实验目录（包含时间戳，避免重名）
args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))  # 创建目录并保存当前脚本（用于复现）
model_path = args.save + '/model_epochs/'  # 模型权重保存路径
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'  # 中间结果图像保存路径
os.makedirs(image_path, exist_ok=True)

# 配置日志：输出到控制台和文件
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))  # 日志文件
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))  # 记录当前脚本名

# 配置张量类型（根据是否使用CUDA）
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # 默认使用CUDA张量
    if not args.cuda:
        print("警告：检测到CUDA设备，但未启用CUDA。使用--cuda可加速训练。")
        torch.set_default_tensor_type('torch.FloatTensor')  # 默认使用CPU张量
else:
    torch.set_default_tensor_type('torch.FloatTensor')  # 无CUDA时使用CPU张量


def save_images(tensor):
    # 将模型输出张量转为可保存的图像格式（同test.py）
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def main():
    # 主训练函数
    if not torch.cuda.is_available():
        logging.info('无可用GPU设备')
        sys.exit(1)

    # 导入混合精度训练相关模块
    from torch.cuda.amp import GradScaler, autocast

    # 初始化混合精度缩放器
    scaler = GradScaler()

    # 设置随机种子（保证结果可复现）
    np.random.seed(args.seed)
    cudnn.benchmark = True  # 启用CUDA卷积优化（输入尺寸固定时加速）
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('使用GPU设备 = %s' % args.gpu)
    logging.info("参数 = %s", args)  # 记录所有参数

    # 初始化模型
    model = Network()  # 实例化主网络
    utils.save(model, os.path.join(args.save, 'initial_weights.pt'))  # 保存初始权重
    # 初始化增强模块的卷积层权重
    model.enhance.in_conv.apply(model.enhance_weights_init)
    model.enhance.conv.apply(model.enhance_weights_init)
    model.enhance.out_conv.apply(model.enhance_weights_init)
    model = model.cuda()  # 转到GPU

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)  # Adam优化器，带权重衰减

    # 计算模型大小（MB）
    MB = utils.count_parameters_in_MB(model)
    logging.info("模型大小 = %f MB", MB)
    print(MB)

    # 加载训练和测试数据集
    train_low_data_names = './data/indoor/dark-1'  # 训练低光图像目录
    TrainDataset = DataLoader(img_dir=train_low_data_names, task='train')  # 训练数据集

    test_low_data_names = './data/indoor/dark-2'  # 测试低光图像目录
    TestDataset = DataLoader(img_dir=test_low_data_names, task='test')  # 测试数据集

    # 创建数据加载器
    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=False, generator=torch.Generator(device='cuda'))  # 训练数据加载器（CUDA生成器）
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False, generator=torch.Generator(device='cuda'))  # 测试数据加载器

    total_step = 0  # 总迭代步数
    model.train()  # 设置为训练模式（启用Dropout、BN更新等）

    # 训练循环
    for epoch in range(args.epochs):
        losses = []  # 记录当前轮次的损失
        # 遍历训练集
        for idx, (input, img_name) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()  # 输入转到GPU，不需要梯度
            optimizer.zero_grad()  # 清空梯度
            optimizer.param_groups[0]['capturable'] = True  # 允许优化器捕获梯度（某些场景需要）

            with autocast():  # 启用混合精度训练
                loss = model._loss(input)  # 计算损失（调用模型的_loss方法）

            scaler.scale(loss).backward()  # 缩放损失并反向传播
            nn.utils.clip_grad_norm_(model.parameters(), 5)  # 梯度裁剪（防止梯度爆炸）
            scaler.step(optimizer)  # 优化器步骤
            scaler.update()  # 更新缩放器

            losses.append(loss.item())  # 记录损失值
            logging.info('train-epoch %03d %03d %f', epoch, idx, loss)  # 记录每步损失

        # 记录当前轮次的平均损失
        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        # 保存当前轮次的模型权重
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

        # 每50轮进行一次测试并保存中间结果
        if epoch % 50 == 0 and total_step != 0:
            model.eval()  # 切换到评估模式
            with torch.no_grad():  # 关闭梯度计算
                for idx, (input, img_name) in enumerate(test_queue):
                    input = Variable(input, volatile=True).cuda()  # 输入转到GPU
                    image_name = img_name[0].split('/')[-1].split('.')[0]  # 提取文件名
                    # 模型推理（输出多个中间结果，用于损失计算和可视化）
                    L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H13_H14_diff, H2_blur, H3_blur = model(
                        input)
                    input_name = '%s' % (image_name)
                    # 处理结果图像
                    H3 = save_images(H3)  # 去噪结果
                    H2 = save_images(H2)  # 增强结果
                    # 创建保存目录
                    os.makedirs(args.save + '/result/denoise/', exist_ok=True)
                    os.makedirs(args.save + '/result/enhance/', exist_ok=True)
                    # 保存结果（含轮次信息，便于观察训练过程）
                    Image.fromarray(H3).save(
                        args.save + '/result/denoise/' + input_name + '_denoise_' + str(epoch) + '.png', 'PNG')
                    Image.fromarray(H2).save(
                        args.save + '/result/enhance/' + input_name + '_enhance_' + str(epoch) + '.png', 'PNG')
            model.train()  # 切换回训练模式

        # 清理未使用的变量和缓存
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()  # 执行主函数