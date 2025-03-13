import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

def conv_and_pool(x, conv):
    x = F.relu(conv(x)).squeeze(3)
    x = F.max_pool1d(x, x.size(2)).squeeze(2)
    return x

class Model(nn.Module):

    """
       <bound method Module.parameters of Model(
        (embedding): Embedding(4762, 300)
        (conv): ModuleList(
            (0): Conv2d(1, 256, kernel_size=(2, 300), stride=(1, 1))
            (1): Conv2d(1, 256, kernel_size=(3, 300), stride=(1, 1))
            (2): Conv2d(1, 256, kernel_size=(4, 300), stride=(1, 1))
        )
        (dropout): Dropout(p=0.5, inplace=False)
        (fc): Linear(in_features=768, out_features=10, bias=True)
        )>
        91.35%
    """
    def __init__(self, config, dataset):
        super(Model, self).__init__()
        self.model_name = 'TextCNN'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name  # 日志
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.batch_size = 128
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.learning_rate = 1e-3  # 学习率
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            vocab = pkl.load(open(config.vocab_path, 'rb'))
            self.n_vocab = len(vocab)
            self.embedding = nn.Embedding(self.n_vocab, config.embed, padding_idx=self.n_vocab - 1)
        self.conv = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, config.embed)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([conv_and_pool(out, conv) for conv in self.conv], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
