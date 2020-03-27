import torch
import torch.nn as nn

import bertfcn
import loaddata

# 数据文件路径
FILE_PATH = 'data/spam.csv'
# 模型参数
WEIGHT_PATH = 'bert-base-uncased'
VOC_PATH = 'bert-base-uncased'

# Dropout率
DROP_OUT = 0.1
# 学习率
LEARNING_RATE = 1e-3

# 训练批次大小
BATCH_SIZE = 16
# 训练轮数
EPOCHS = 10
# 训练设备
DEVICE = torch.device('cuda:0')


# 训练函数
def train(net, loader, criterion, optimizer):
    # 遍历每个batch
    for batch_idx, (data, target) in enumerate(loader):
        # 如果计算机有NVIDIA GPU，则将数据放在GPU上训练
        if torch.cuda.is_available():
            data, target = data.to(DEVICE), target.to(DEVICE)
        # 得到模型输入序列，大小为[batch_size, sequence_length]
        input_seq = data.squeeze(dim=1)
        # 得到标签序列，大小为[batch_size]
        target = target.squeeze()
        print(batch_idx)
        # 正向传播得到结果
        out = net(input_seq)
        # 清零梯度
        optimizer.zero_grad()
        # 计算loss
        loss = criterion(out, target)
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
        # 每5个batch输出一次训练情况
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


# 验证函数，注意验证过程不需要反向传播
def eval(net, loader, criterion):
    val_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        # 如果计算机有NVIDIA GPU，则将数据放在GPU上验证
        if torch.cuda.is_available():
            data, target = data.to(DEVICE), target.to(DEVICE)
        input_seq = data.squeeze(dim=1)
        target = target.squeeze()
        # 正向传播得到结果
        out = net(input_seq)
        # 累加loss
        val_loss += criterion(out, target).item()
        # 将输出值最大的一项作为预测结果
        pred = out.argmax()
        # 累加预测正确的数目
        correct += pred.eq(target.data).float().sum().item()

    # 计算平均loss
    val_loss /= len(val_dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss, correct, len(val_dataset),
                                                                                100. * correct / len(val_dataset)))


# 加载数据
content_list, label_list = loaddata.read_data(FILE_PATH, VOC_PATH)
# 建立数据集
all_dataset = loaddata.SpamLabelDataset(content_list, label_list)
# 将数据集划分为训练集、验证集和测试集
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [4000, 1000, 572])

# 建立批处理抽取器
train_loader = loaddata.set_dataloader(train_dataset, BATCH_SIZE)
val_loader = loaddata.set_dataloader(val_dataset, BATCH_SIZE)
test_loader = loaddata.set_dataloader(test_dataset, BATCH_SIZE)

# 实例化模型
model = bertfcn.BertFcModel(WEIGHT_PATH, DROP_OUT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 如果计算机有NVIDIA GPU，则将模型放在GPU上训练
if torch.cuda.is_available():
    model.to(DEVICE)

# 开始训练
for epoch in range(EPOCHS):
    # 将模型切换为训练模式
    model.train()
    train(model, train_loader, criterion, optimizer)
    # 将模型切换为验证模式（在验证模式中模型不会进行Dropout操作）
    model.eval()
    # 每训练一个epoch对模型性能进行一次验证
    eval(model, val_loader, criterion)
