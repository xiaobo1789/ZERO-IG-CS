import os
import sys
import numpy as np
import torch
import argparse
import logging
import torch.utils
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel  # 导入微调模型类
from multi_read_data import DataLoader  # 导入自定义数据加载器
from thop import profile  # 用于计算模型FLOPs


# 设置根目录并添加到系统路径（便于模块导入）
root_dir = os.path.abspath('../')
sys.path.append(root_dir)

# 解析命令行参数
parser = argparse.ArgumentParser("ZERO-IG")  # 定义参数解析器，描述为"ZERO-IG"
parser.add_argument('--data_path_test_low', type=str, default='./data',
                    help='测试低光图像数据路径')
parser.add_argument('--save', type=str,
                    default='./results/',
                    help='结果保存路径')
parser.add_argument('--model_test', type=str,
                    default='./model',
                    help='预训练模型路径')
parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
parser.add_argument('--seed', type=int, default=2, help='随机种子（保证结果可复现）')

args = parser.parse_args()  # 解析参数
save_path = args.save
os.makedirs(save_path, exist_ok=True)  # 创建结果保存目录，已存在则不报错

# 配置日志：输出到控制台和文件
log_format = '%(asctime)s %(message)s'  # 日志格式：时间+消息
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')  # 配置控制台日志
mertic = logging.FileHandler(os.path.join(args.save, 'log.txt'))  # 日志文件路径
mertic.setFormatter(logging.Formatter(log_format))  # 设置文件日志格式
logging.getLogger().addHandler(mertic)  # 将文件日志添加到日志器

logging.info("train file name = %s", os.path.split(__file__))  # 记录当前脚本文件名

# 加载测试数据集
TestDataset = DataLoader(img_dir=args.data_path_test_low, task='test')  # 实例化数据集
test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)  # 数据加载器（批大小1，不打乱）


def save_images(tensor):
    # 将模型输出的张量转为可保存的图像格式
    image_numpy = tensor[0].cpu().float().numpy()  # 取第0个样本，转到CPU，转为numpy
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  # 调整维度：(C,H,W)→(H,W,C)
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')  # 反归一化（乘255），裁剪到[0,255]，转为uint8
    return im


def calculate_model_parameters(model):
    # 计算模型总参数量
    return sum(p.numel() for p in model.parameters())  # 遍历所有参数，求和每个参数的元素数


def calculate_model_flops(model, input_tensor):
    # 计算模型FLOPs（浮点运算次数）
    flops, _ = profile(model, inputs=(input_tensor,))  # 使用thop库计算
    flops_in_gigaflops = flops / 1e9  # 转换为GigaFLOPs（1e9）
    return flops_in_gigaflops


def main():
    # 主测试函数
    if not torch.cuda.is_available():
        print('无可用GPU设备')
        sys.exit(1)

    # 加载预训练模型
    model = Finetunemodel(args.model_test)  # 实例化微调模型，加载预训练权重
    model = model.cuda()  # 转到GPU
    model.eval()  # 设置为评估模式（关闭Dropout等）

    # 计算模型参数量
    total_params = calculate_model_parameters(model)
    print("总参数量: ", total_params)

    # 冻结模型参数（测试时不更新）
    for p in model.parameters():
        p.requires_grad = False

    # 无梯度计算（加速测试，节省内存）
    with torch.no_grad():
        # 遍历测试集
        for _, (input, img_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()  # 包装为Variable并转到GPU（volatile=True：不追踪梯度）
            input_name = img_name[0].split('/')[-1].split('.')[0]  # 提取文件名（不含路径和后缀）
            enhance, output = model(input)  # 模型推理：得到增强图和去噪图

            # 处理并保存结果
            enhance = save_images(enhance)  # 增强图转为图像格式
            output = save_images(output)  # 去噪图转为图像格式
            os.makedirs(args.save + '/result', exist_ok=True)  # 创建结果保存目录
            Image.fromarray(output).save(args.save + '/result/' + input_name + '_denoise' + '.png', 'PNG')  # 保存去噪图
            Image.fromarray(enhance).save(args.save + '/result/' + input_name + '_enhance' + '.png', 'PNG')  # 保存增强图

    torch.set_grad_enabled(True)  # 恢复梯度计算（可选，测试结束后不影响）


if __name__ == '__main__':
    main()  # 执行主函数