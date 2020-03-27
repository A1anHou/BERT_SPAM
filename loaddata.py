import pandas as pd
from torch.utils.data import DataLoader, dataset
import torch
from transformers import BertTokenizer


# 从文件中读取数据
def read_data(file_path, voc_path):
    # 从文件中读取数据
    data_corpus = pd.read_csv(file_path)
    # 获取邮件内容数据并转换为字符串列表
    content_list = list(data_corpus['v2'].values)
    # 加载Bert分词工具
    tokenizer = BertTokenizer.from_pretrained(voc_path)
    # 将邮件内容编码为词表中对应的数字，并将输入长度统一为256
    for i in range(len(content_list)):
        content_list[i] = tokenizer.encode(content_list[i], max_length=128, pad_to_max_length=True)
    # 获取标签数据并转换为字符串列表
    label_list = list(data_corpus['v1'].values)
    # 将标签编码为数字，是垃圾邮件编码为1，不是垃圾邮件编码为0
    for i in range(len(label_list)):
        label_list[i] = 0 if label_list[i] == 'ham' else 1
    return content_list, label_list


# 建立数据集
class SpamLabelDataset(dataset.Dataset):
    def __init__(self, content_list, label_list):
        self.content_list = content_list
        self.label_list = label_list

    def __len__(self):
        # 返回数据长度
        return len(self.content_list)

    def __getitem__(self, ind):
        content = self.content_list[ind]
        label = self.label_list[ind]
        # 将数据转换为tensor类型
        content = torch.LongTensor([content])
        label = torch.LongTensor([label])
        return content, label


# 建立批处理抽取器
def set_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)
    return dataloader
