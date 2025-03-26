# coding: UTF-8
import torch.nn as nn
import pickle as pkl

class Model(nn.Module):

    def __init__(self, config, dataset):
        super(Model, self).__init__()
        self.model_name = 'TextRNN'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name  # 日志
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.n_vocab = 0  # 词表大小，运行时赋值
        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            vocab = pkl.load(open(config.vocab_path, 'rb'))
            self.n_vocab = len(vocab)
            self.embedding = nn.Embedding(self.n_vocab, config.embed, padding_idx=self.n_vocab - 1)
        # LSTM双向结构图
        self.lstm = nn.LSTM(config.embed, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x  # 解包输入
        out = self.embedding(x)  # 词嵌入层 词索引->词向量  [batch_size, seq_len, embedding]=[128, 32, 300]
        out, _ = self.lstm(out)  # LSTM处理
        out = self.fc(out[:, -1, :])  # 取最后时间步的输出，传入全连接层
        return out
