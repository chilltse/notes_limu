import torchvision
from torchvision import transforms
from torch.utils import data
import torch

import torch

def try_all_gpus():
    """返回所有可用的GPU设备（如果没有，则返回 [cpu]）"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def get_dataloader_workers():
    """使用4个进程来读取数据。"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """加载Fashion-MNIST数据集，分为训练集和测试集。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='../',
                                                  train=True,
                                                  transform=trans,
                                                  download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../',
                                                  train=False,
                                                  transform=trans,
                                                  download=True)

    return (data.DataLoader(mnist_train, batch_size, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, num_workers=get_dataloader_workers()))




class Accumulator:
    """在n个变量上累加。"""
    def __init__(self, n):
        # 创建一个长度为 n 的列表，比如：
        # n = 2 → [0.0, 0.0]
        self.data = [0.0] * n

    def add(self, *args):
        # zip(self.data, args)：
        # 把两个列表打包成一一对应的对：
        # [0.0, 0.0], (3, 7) → [(0.0, 3), (0.0, 7)]
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # ✅ y_hat = y_hat.argmax(axis=1)
        # 对每一行取最大值的索引（也就是预测的类别）：
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        #
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 用来累加 [正确预测数, 总样本数]

    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())  # 添加正确数和样本数

    return metric[0] / metric[1]  # 返回准确率 = 正确 / 总数

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）。"""
    if isinstance(net, torch.nn.Module):
        net.train()  # 设置为训练模式
    metric = Accumulator(3)  # [累加loss, 预测正确数, 样本总数]

    for X, y in train_iter:
        y_hat = net(X)              # 前向传播
        l = loss(y_hat, y)          # 计算损失

        # 标准 PyTorch 优化器
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()     # 清空梯度
            l.backward()            # 反向传播
            updater.step()          # 更新参数
            # 你用的是 PyTorch 自带的 nn.CrossEntropyLoss() 或类似的 loss，它内部已经做了 .mean()，返回的是一个 标量张量。
            metric.add(
                float(l) * len(y),              # 累加loss（平均loss * 样本数）
                accuracy(y_hat, y),             # 累加正确预测数量
                y.numel()                       # 累加样本数
            )
        else:
            # 自定义优化器逻辑
            l.sum().backward()
            updater(X.shape[0])
            metric.add(
                float(l.sum()),
                accuracy(y_hat, y),
                y.numel()
            )
    # 第一个是 所有的loss累加/样本总数； 第二个是正确率
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均loss和准确率


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        # 训练一轮，返回平均 loss 和准确率
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)

        print(f"{train_metrics = }")

        # 在测试集上评估准确率
        test_acc = evaluate_accuracy(net, test_iter)

        print(f"{test_acc = }")


from torch.utils import data

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 PyTorch 数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 造数据，用的是正态分布，并加入了正态分布的噪音
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # print(X.shape)
    Y = torch.matmul(X, w) + b
    Y += torch.normal(0, 0.01, Y.shape)
    return X, Y.reshape((-1, 1))

def linreg(X,w,b):
    return X @ w + b

def squared_loss(y_hat, y):
    """"均方误差"""
    # print(((y_hat - y.reshape(y_hat.shape)) ** 2/ 2))
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失。"""
    metric = Accumulator(2)  # [累计的 loss 总和, 样本总数]

    for X, y in data_iter:
        out = net(X)                          # 预测值
        y = y.reshape(out.shape)              # 调整 y 的形状与 out 匹配
        l = loss(out, y)                      # 计算损失张量（每个样本都有一个）

        metric.add(l.sum(), l.numel())        # 累加总损失 和 样本数

    return metric[0] / metric[1]              # 返回平均损失

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            # 这里的/ batch_size其实是一个标量，让learning rate不随着batch size的变化而变化
            param -= lr * param.grad / batch_size
            param.grad.zero_()

import matplotlib.pyplot as plt
from IPython import display
from matplotlib_inline.backend_inline import set_matplotlib_formats


# 配置坐标轴函数
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴属性"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# 设置 SVG 输出
def use_svg_display():
    """使用svg格式在Jupyter中显示图像（高清）"""
    set_matplotlib_formats('svg')


# 动态绘图类
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """在一个图里动态绘制多条线"""
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """添加数据点并更新图像"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        if not hasattr(x, "__len__"):
            x = [x] * n
        for i, (a, b) in enumerate(zip(x, y)):
            self.X[i].append(a)
            self.Y[i].append(b)
        self.axes[0].cla()  # 清空当前 axes
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


import torch

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)
            y_train = torch.cat([y_train, y_part], dim=0)
    return X_train, y_train, X_valid, y_valid


# def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
#     train_l_sum, valid_l_sum = 0, 0
#     for i in range(k):
#         data = get_k_fold_data(k, i, X_train, y_train)  # 拿到第 i 折的数据
#         net = get_net()  # 初始化一个新网络（避免参数被上一轮训练改变）
#         train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
#
#         train_l_sum += train_ls[-1]  # 取训练最后一轮的损失
#         valid_l_sum += valid_ls[-1]  # 取验证集最后一轮的损失
#
#         if i == 0:  # 第一折，画图
#             plt.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
#                      xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
#                      legend=['train', 'valid'], yscale='log')
#
#         print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):.6f}, '
#               f'valid log rmse {float(valid_ls[-1]):.6f}')
#
#     return train_l_sum / k, valid_l_sum / k  # 返回 k 折平均训练损失和验证损失

# 计算精度
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # 累计正确数量、总数量
    for X, y in data_iter:
        if isinstance(X, list):  # 兼容多个输入
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        # 累加：预测正确数量，标签总数量
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

import time
import numpy as np
from torch import nn

class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并记录时间。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回总时间。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

from torch import nn

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6 of D2L)"""

    # 权重初始化函数（适用于Linear和Conv2D）
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # 这种初始化是为了保证数值稳定性，保证均值为0方差差不多
            nn.init.xavier_uniform_(m.weight)

    # 应用初始化
    net.apply(init_weights)

    print('training on', device)
    net.to(device)

    # 优化器 & 损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # 绘图工具
    # animator = Animator(
    #     xlabel='epoch', xlim=[1, num_epochs],
    #     legend=['train loss', 'train acc', 'test acc']
    # )

    # 定时器 & 批次数
    timer, num_batches = Timer(), len(train_iter)

    for epoch in range(num_epochs):
        metric = Accumulator(3)  # 累加 train_loss, train_acc_sum, sample_count
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # animator.add(
                #     epoch + (i + 1) / num_batches,
                #     (metric[0] / metric[2], metric[1] / metric[2], None)
                # )

        # 每轮结束后在 test 数据集上评估准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # animator.add(epoch + 1, (None, None, test_acc))

    # 打印最终结果
    print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

import torch

def try_gpu(i=0):
    """如果存在，则返回 GPU(i)，否则返回 CPU。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


import matplotlib.pyplot as plt


def set_figsize(figsize=(3.5, 2.5)):
    """设置默认图像尺寸"""
    plt.rcParams['figure.figsize'] = figsize

import matplotlib.pyplot as plt
from torchvision import transforms

# 显示图像列表
def show_images(imgs, num_rows, num_cols, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)  # 图像整体尺寸
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)  # 创建子图网格
    axes = axes.flatten()  # 将二维坐标轴变成一维，方便迭代
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if isinstance(img, torch.Tensor):
            # 把形状从 (C, H, W) 转成 (H, W, C)，用于 matplotlib 显示
            img = img.permute(1, 2, 0).detach().numpy()
        ax.imshow(img)           # 显示图像
        ax.axes.get_xaxis().set_visible(False)  # 不显示x轴刻度
        ax.axes.get_yaxis().set_visible(False)  # 不显示y轴刻度
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


def imshow(img):
    plt.imshow(img)
    plt.axis('off')  # 可选：不显示坐标轴
    plt.title('Example Image')  # 可选：加标题
    plt.show()




#########Models##########
import torch
from torch import nn
from torch.nn import functional as F

# ---------------------
# 定义基本残差块 Residual
# ---------------------
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# ---------------------
# 构建ResNet基本层 block
# ---------------------
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# ---------------------
# 构建 ResNet-18 网络
# ---------------------
def resnet18(num_classes=10, in_channels=1):  # ← 增加第二个参数 in_channels
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # ← 使用 in_channels
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module("resnet_block1", nn.Sequential(*resnet_block(64, 64, 2, first_block=True)))
    net.add_module("resnet_block2", nn.Sequential(*resnet_block(64, 128, 2)))
    net.add_module("resnet_block3", nn.Sequential(*resnet_block(128, 256, 2)))
    net.add_module("resnet_block4", nn.Sequential(*resnet_block(256, 512, 2)))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("flatten", nn.Flatten())
    net.add_module("fc", nn.Linear(512, num_classes))
    return net


