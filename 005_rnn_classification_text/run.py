import argparse
from enum import Enum
import torch
from importlib import import_module
import numpy as np
import time
from tensorboardX import SummaryWriter

from dataset_Iterator import build_iterator
from train_eval import init_network, train
from utils import build_dataset, get_time_dif

# 通过命令行的方式指定参数
parser = argparse.ArgumentParser(description="Classification Text")
parser.add_argument('--model', type=str, required=True, help="choose model：Text_CNN, Text_RNN")
parser.add_argument('--embedding', default='sogou', type=str, help='random or sogou、tencent')
args = parser.parse_args()

class Embedding(Enum):
    sou_gou_embedding = "embedding_SougouNews.npz"  # 搜狗新闻
    tencent_embedding = "embedding_Tencent.npz"  # 腾讯
    random_embedding = "random"  # 随机初始化

class SourceConfig(object):
    def __init__(self, dataset, _embedding):
        self.train_path = dataset + '/data/train.txt'  # 训练集路径
        self.dev_path = dataset + '/data/dev.txt'  # 验证集路径
        self.test_path = dataset + '/data/test.txt'  # 测试集路径
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]  # 分类类别
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表路径
        self.num_classes = len(self.class_list)  # 类别个数
        self.embedding_pretrained = (torch.tensor(  # 词向量
            np.load(dataset + '/data/' + _embedding)["embeddings"].astype('float32'))
            if _embedding != 'random' else None  # random返回None
        )  # 词向量
        self.embed = (  # 字向量维度, 若使用了预训练词向量，则维度统一
            self.embedding_pretrained.size(1)
            if self.embedding_pretrained is not None else 300  # 等于None返回300
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备类型

def keep_seed():
    # 固定种子，保证在运行时的随机性和计算过程是可重复的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    dataset_name = 'THUCNews'  # 数据集

    # args
    model_name = args.model
    embedding = args.embedding
    # args model:Text_RNN, embedding:tencent
    print(f"args model:{model_name}, embedding:{embedding}")

    if args.embedding == 'sougou':
        embedding = Embedding.sou_gou_embedding.value
    elif args.embedding == "tencent":
        embedding = Embedding.tencent_embedding.value
    else:
        embedding = Embedding.random_embedding

    source_config = SourceConfig(dataset_name, embedding)
    model = import_module('models.' + model_name).Model(source_config, dataset_name)

    keep_seed()

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(model, source_config)

    train_iter = build_iterator(train_data, model.batch_size, source_config.device)
    dev_iter = build_iterator(dev_data, model.batch_size, source_config.device)
    test_iter = build_iterator(test_data, model.batch_size, source_config.device)
    time_dif = get_time_dif(start_time)
    print(f"Time usage:{time_dif}")

    num_vocab = len(vocab)
    writer = SummaryWriter(log_dir=model.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    init_network(model)
    train(source_config, model, train_iter, dev_iter, test_iter, writer)
