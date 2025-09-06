import torchvision
from torchvision import transforms
from torch.utils import data
import torch

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


def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 对应区域相乘后求和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


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


import matplotlib.pyplot as plt
from IPython import display
from matplotlib_inline.backend_inline import set_matplotlib_formats

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

#########################TIME SEQUENCE#######################################

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符标记。"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知令牌类型：' + token)


import collections

def count_corpus(tokens):
    """统计标记的频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将二维嵌套列表展平
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    """文本词汇表"""
    # min_freq表示一个阈值，如果出现的次数少于它的话就统一归类成"unknown"
    # reserved_tokens: 保留符号，如 <pad>、<bos>、<eos> 等特殊token。
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        # 所以这个 lambda 函数表示：按照每个元素的第 2 项（即词频）排序;reverse=True表示降序排列（从大到小），默认是升序。
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # unk表示unknown;
        #     <unk>	处理低频或未知词汇
        #     <pad>	填充序列对齐长度
        #     <bos>	序列起始（begin of sentence）
        #     <eos>	序列结束（end of sentence）
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]

        self.idx_to_token, self.token_to_idx = [], {}
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1


    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]



import re
import requests

# 如果本地没有文件就下载
import os

# 加载并清洗文本
def read_time_machine():
    """Load the time machine dataset into a list of cleaned text lines."""
    with open(r"F:\000-CS\LIMU_BILIBILI\timemachine.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把不是大小写字母的东西全部变成空格
    # re.sub(pattern, replacement, string)：表示用 replacement 替换 string 中所有匹配 pattern 的部分。
    # .strip()去除处理后字符串的首尾空格。
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 主函数：加载语料和词汇表
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的标记索引列表和词汇表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    # tokens = tokenize(lines, 'word')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]  # 展平为 index 序列
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

import random
import torch

# 读取文本并进行分词（返回的是一个二维列表，每行是一个句子的词元列表）
tokens = tokenize(read_time_machine())
# 将二维词元列表展平成一维的词元序列
corpus = [token for line in tokens for token in line]
# 构建词汇表
vocab = Vocab(corpus)
# 查看前10个词元及其频率
vocab.token_freqs[:10]


# 随机采样：一个batch中，子序列之间是无序的
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签，即在最后一个组别的时候，我们也可以通过+1去找到它对应的y
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    # 表示从 start 开始，到 stop 之前（不包括stop），每次增加 step 的一个序列。
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 顺序分区：一个batch中，子序列之间也是顺序产生的
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


