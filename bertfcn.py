from transformers import BertModel
import torch.nn as nn
import torch


# Bert+FCN模型
class BertFcModel(nn.Module):
    def __init__(self, weight_path, drop_out):
        super(BertFcModel, self).__init__()
        # 加载Bert模型
        self.model = BertModel.from_pretrained(weight_path)
        # 全连接层，将Bert输出的768维向量转换为1维
        self.fcn = nn.Linear(768, 2)
        # 指定Bert输出与全连接层输入之间的Dropout率
        self.dropout = nn.Dropout(drop_out)

    # 前向传播过程
    def forward(self, input_seq):
        out = self.model(input_seq)
        # 将[CLS]标签对应的Bert最后一层输出作为全连接网络的输入
        out = out[0][:, 0, :]
        out = self.dropout(out)
        out = self.fcn(out)
        # 为输出计算SoftMax
        out = torch.softmax(out, dim=1)
        return out
