import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from pytorch_grad_cam import GradCAM

import argparse
from time import time
import math

import torch
import torch.nn as nn
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src import EVT as cct_models
from utils.losses import LabelSmoothingCrossEntropy
import torch.nn.functional as F

model_names = sorted(name for name in cct_models.__dict__
                     if name.islower() and not name.startswith("_")
                     and callable(cct_models.__dict__[name]))
import os
import random
import cv2
from PIL import Image


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='log frequency (by iteration)')

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=5, type=int, metavar='N',
                        help='number of warmup epochs')
    ###
    parser.add_argument('-b', '--batch-size', default=6, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')

    '''parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')'''

    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--clip-grad-norm', default=0., type=float,
                        help='gradient norm clipping (default: 0 (disabled))')

    parser.add_argument('-m', '--model',
                        type=str.lower,
                        choices=model_names,
                        default='cct_2', dest='model')

    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='learnable', dest='positional_embedding')

    parser.add_argument('--conv-layers', default=3, type=int,
                        help='number of convolutional layers (cct only)')

    parser.add_argument('--conv-size', default=3, type=int,
                        help='convolution kernel size (cct only)')

    parser.add_argument('--patch-size', default=4, type=int,
                        help='image patch size (vit and cvt only)')

    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')

    parser.add_argument('--disable-aug', action='store_true',
                        help='disable augmentation policies for training')

    parser.add_argument('--gpu_id', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable cuda')

    return parser


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45, fontsize=14)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=14)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels', fontsize=14)
        plt.ylabel('Predicted Labels', fontsize=14)
        plt.title('EVT+wpe+nie', fontsize=16)

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black", fontsize=14)
        plt.tight_layout()
       
        plt.savefig('evt_wpe_eni.png', dpi=300)
        plt.show()

 # 绘制所有图像的热力图
def plot_image_grid(images, figsize=(15, 15)):
    num_images = len(images)
    if num_images == 0:
        print("没有图像可供显示。")
        return
    
    # 动态计算网格的行数和列数，尽量接近正方形
    cols = int(math.ceil(math.sqrt(num_images)))  # 列数为图像数量的平方根的向上取整
    rows = int(math.ceil(num_images / cols))  # 行数根据列数计算

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]  # 如果只有一个子图，将其转换为列表
    axes = axes.flatten()  # 展平为一维数组

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.axis('off')
        else:
            ax.axis('off')  # 如果图像数量不足，关闭多余的子图
    plt.tight_layout()
    plt.savefig("./show_test4.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def reshape_transform(tensor, height=None, width=None):
    # 去掉 CLS Token（如果存在）
    tensor = tensor[:, 1:, :]

    # 动态计算 height 和 width
    if height is None or width is None:
        seq_length = tensor.size(1)
        hidden_dim = tensor.size(2)
        # 假设 seq_length 是 height × width
        height = int(seq_length ** 0.5)
        width = seq_length // height

    # 重塑张量
    result = tensor.reshape(tensor.size(0), height, width, hidden_dim)
    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)  # (B, C, H, W)
    return result

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), r"test"))  # get data root path
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "/moak/mount/qycode/data"))
    # image_path = os.path.join(data_root, "breakhis_test")  # flower data set path
    # assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "/moak/mount/qycode/data/")) 

    validate_dataset = datasets.ImageFolder(root=os.path.join(data_root, "bhzq"),
                                            transform=data_transform)
    
    # for img_path, label in validate_dataset.samples:
    #     file_name = os.path.basename(img_path)
    #     print(file_name, label)
    # exit()
    batch_size = 1
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    parser = init_parser()
    args = parser.parse_args()
    model = cct_models.__dict__[args.model](img_size=224,
                                            num_classes=2,
                                            positional_embedding=args.positional_embedding,
                                            n_conv_layers=args.conv_layers,
                                            kernel_size=args.conv_size,
                                            patch_size=args.patch_size)
    # load pretrain weights
    model_weight_path = "evt-bhzq-1015.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    model.eval()
    # model.tokenizer.conv_layers[0][0]较早的卷积层 
    # model.tokenizer.conv_layers[1][0]和model.tokenizer.conv_layers[2][0]是中间层卷积 
    # model.classifier.attention_pool注意力池化 
    # transformer编码器层 model.classifier.blocks[0]  model.classifier.blocks[1]
    target_layer = model.tokenizer.conv_layers[2][2]
    all_superimposed_imgs = []
    # with torch.no_grad():
    for i, val_data in enumerate(validate_loader):
        val_images, val_labels = val_data
        val_images.requires_grad_(True).to(device)
        
        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(val_labels.item())]  # 替换为目标类别索引
        grayscale_cam = cam(input_tensor=val_images, targets=targets)[0, :]
        
        # 将灰度热力图转换为彩色热力图
        heatmap = np.uint8(255 * grayscale_cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 加载原始图像
        img_path = validate_dataset.samples[i][0]  # 获取当前图像路径
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # 调整大小以匹配热力图
        
        # 叠加热力图到原始图像
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        cv2.imwrite('test4_{}.png'.format(i), superimposed_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9, int(cv2.IMWRITE_PNG_BILEVEL), 0])
        
        # 将结果保存到列表中
        all_superimposed_imgs.append(superimposed_img)

        # 打印模型输出和标签
        outputs = model(val_images.to(device))
        outputs = torch.softmax(outputs, dim=1)
        print(outputs.tolist()[0][0], outputs.tolist()[0][1], val_labels.tolist()[0])
        outputs = torch.argmax(outputs, dim=1)
        confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())

   

    # 调用函数绘制图像网格
    plot_image_grid(all_superimposed_imgs)
    confusion.plot()
    confusion.summary()

