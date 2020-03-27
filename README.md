

# 基于BERT的垃圾邮件分类（bert-spam）

### 环境配置

- 如果未安装``` pyTorch``` 请先安装
- 运行```pip install transformers``` 安装 [huggingface/transformers](https://github.com/huggingface/transformers)

### 文件说明

- ```data\spam.csv``` 为垃圾邮件分类数据文件
- ```loaddata.py``` 为加载数据模块，```bertfcn.py``` 为模型定义模块， ```main.py``` 为主程序入口

### 运行说明

- 可以使用 ``` python main.py``` 命令运行 ``` main.py``` 或直接在IDE中运行 ``` main.py```


### 数据来源

https://www.kaggle.com/uciml/sms-spam-collection-datasetxian

### 参考文献

[1] Pytorch官方文档 <https://pytorch-cn.readthedocs.io/zh/latest/>

[2] Transformers官方文档 <https://huggingface.co/transformers/>

[3] pytorch-transformers （BERT）微调. wenqiang su <https://blog.csdn.net/weixin_42681868/article/details/102538980>