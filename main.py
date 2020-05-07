import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split#将数据分为测试集和训练集
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import trange
from torch import optim

# 数据文件路径
FILE_PATH = 'data/spam.csv'
# 模型和词表路径
WEIGHT_PATH = 'data/bert_base_uncased'
VOC_PATH = 'data/bert_base_uncased/bert-base-uncased-vocab.txt'
DEVICE = torch.device('cuda:0')

# 读取训练数据 os.path.join(data_dir, "train.txt")
df = pd.read_csv(FILE_PATH)
# 提取语句并处理
labels = [0 if label == 'ham' else 1 for label in df.v1.values]
print(df.v2.values[0])
tokenizer = BertTokenizer.from_pretrained(VOC_PATH, do_lower_case=True)
tokenized_sents = [tokenizer.encode(sent, max_length=128, pad_to_max_length=True) for sent in df.v2.values]

#划分训练集、验证集
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(tokenized_sents, labels,
                                                            random_state=2018, test_size=0.1)

print("训练集的一个inputs",tokenizer.decode(train_inputs[0]))

#将训练集、验证集转化成tensor
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

#生成dataloader
batch_size = 32
train_data = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs,  validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained(WEIGHT_PATH)
print(model.cuda())


# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'gamma', 'beta']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.0}
# ]

# optimizer = AdamW(optimizer_grouped_parameters,
#                      lr=2e-5)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

#定义一个计算准确率的函数
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#训练开始
epochs = 10
for _ in trange(epochs, desc="Epoch"):
    #训练开始
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(DEVICE) for t in batch)
        b_input_ids, b_labels = batch
        optimizer.zero_grad()
        #取第一个位置，BertForSequenceClassification第一个位置是Loss，第二个位置是[CLS]的logits
        loss = model(b_input_ids, token_type_ids=None,  labels=b_labels)[0]
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    #模型评估
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        b_input_ids, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None)[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

