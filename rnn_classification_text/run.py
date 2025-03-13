import argparse
from enum import Enum
import torch
from importlib import import_module
import numpy as np
import time
from tensorboardX import SummaryWriter

from rnn_classification_text.dataset_Iterator import build_iterator
from rnn_classification_text.train_eval import init_network, train
from rnn_classification_text.utils import build_dataset, get_time_dif

# 通过命令行的方式指定参数
parser = argparse.ArgumentParser(description="Classification Text")
parser.add_argument('--model', type=str, required=True, help="choose model：Text_CNN, Text_RNN")
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
args = parser.parse_args()

class Embedding(Enum):
    sou_gou_embedding = "embedding_SougouNews.npz"  # 搜狗新闻
    tencent_embedding = "embedding_Tencent.npz"  # 腾讯
    random_embedding = "random"  # 随机初始化

class SourceConfig(object):
    def __init__(self, dataset, _embedding):
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]  # 分类类别
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.num_classes = len(self.class_list)  # 类别数
        self.embedding_pretrained = (torch.tensor(
            np.load(dataset + '/data/' + _embedding)["embeddings"].astype('float32'))
            if _embedding != 'random' else None  # 等于random返回None
        )  # 词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备类型
        self.embed = (
            self.embedding_pretrained.size(1)
            if self.embedding_pretrained is not None else 300  # 等于None返回300
        )  # 字向量维度, 若使用了预训练词向量，则维度统一

def keep_seed():
    # 固定种子，保证在运行时的随机性和计算过程是可重复的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    """
    循环神经网（Recurrent Neural Network，RNN）
    文本 -> 文本数据清洗 -> 分字转id（pkl） -> 字映射向量(npz)
    
    金证顾问：过山车行情意味着什么
    [18, 249, 1086, 438, 4, 268, 169, 121, 46, 143, 274, 1342, 1068, 1046, 1081, 4760, 4760, 4760, 4760, 4760, 4760, 
    4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760]
    tensor([[  18,  249, 1086,  ..., 4760, 4760, 4760],...])
    tensor([[[ 3.0235e-01,  2.0894e-01, -8.0932e-02,  ..., -4.3194e-02,-3.1051e-01,  1.8790e-01],...]])

    
    
    
    金证顾问：过山车行情意味着什么	2
    
    词索引列表, 标签, 序列长度
    [([18, 249, 1086, 438, 4, 268, 169, 121, 46, 143, 274, 1342, 1068, 1046, 1081, 4760, 4760, 4760, 4760, 4760, 4760, 
    4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760, 4760], 2, 15)]    
    
    to tensor
    ((
    tensor([[  18,  249, 1086,  ..., 4760, 4760, 4760],
        [  14,  125,   55,  ..., 4760, 4760, 4760],
        [ 135,   80,   33,  ..., 4760, 4760, 4760],
        ...,
        [ 160, 1667, 1147,  ..., 4760, 4760, 4760],
        [  31,   75,    4,  ..., 4760, 4760, 4760],
        [ 321,  566,  130,  ..., 4760, 4760, 4760]]), 
    tensor([15, 18, 22, 25, 25, 23, 20, 17, 22, 16, 11, 23, 23, 22,  7, 23, 20, 25,
        15,  9, 17, 15, 24, 20, 17, 17, 13, 20, 19, 20, 22, 22, 21, 22, 23, 19,
        12, 20, 23, 18, 22, 25, 23, 20, 19, 17, 17, 15, 17, 26, 16, 22, 21, 18,
        16, 12, 23, 19, 20, 21, 12, 24, 18, 14, 25, 16, 24, 24, 23, 20, 20, 20,
        18, 16, 23, 14, 23, 21, 19, 17, 24, 21, 23, 23, 19, 15, 12, 22, 25, 14,
        21, 20, 22, 15, 22, 18, 16, 17, 13, 21, 21, 18, 21, 11, 19, 22, 14, 22,
        15, 22, 15, 22, 22, 15, 25, 16, 18, 18, 14, 19, 13, 29, 20, 18, 22, 16,
        18, 22])), 
    tensor([2, 3, 4, 1, 7, 5, 5, 9, 1, 8, 4, 3, 7, 5, 1, 8, 1, 1, 8, 4, 4, 6, 7, 1,
        9, 4, 2, 9, 4, 2, 2, 9, 8, 9, 1, 3, 9, 5, 9, 6, 7, 2, 9, 5, 9, 4, 5, 6,
        8, 1, 2, 1, 4, 0, 5, 4, 9, 6, 5, 5, 2, 4, 5, 5, 7, 8, 6, 7, 7, 2, 9, 0,
        4, 6, 7, 2, 9, 7, 9, 0, 2, 9, 9, 4, 9, 0, 0, 4, 1, 2, 5, 5, 7, 0, 5, 9,
        5, 3, 4, 6, 8, 3, 5, 9, 3, 9, 4, 9, 5, 4, 6, 2, 3, 6, 7, 4, 6, 2, 2, 2,
        0, 1, 6, 4, 4, 2, 2, 3]
    ))
    
    字映射向量
    tensor([[[ 3.0235e-01,  2.0894e-01, -8.0932e-02,  ..., -4.3194e-02,
          -3.1051e-01,  1.8790e-01],
         [ 3.7446e-02, -5.7123e-02, -2.5790e-01,  ..., -2.9264e-01,
           1.8909e-01, -5.4846e-01],
         [-2.5890e-02,  1.3263e-01, -4.0175e-01,  ...,  3.4654e-01,
          -5.0803e-01, -1.8250e-01],
         ...,
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01]],

        [[ 1.0809e-01,  8.7689e-02, -2.3905e-01,  ..., -4.2419e-01,
           6.5307e-02, -1.2141e-01],
         [-1.5821e-01,  5.3195e-01,  1.8578e-01,  ..., -1.6736e-01,
          -2.8041e-01,  1.5411e-01],
         [ 4.3293e-02,  6.1321e-01, -1.0018e-02,  ...,  6.1660e-02,
          -3.4322e-01,  9.0101e-02],
         ...,
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01]],

        [[ 3.1487e-01, -3.2435e-01,  1.3675e-01,  ...,  1.9030e-01,
           1.3956e-01,  7.8458e-02],
         [-1.5683e-02,  9.9436e-02, -4.0968e-01,  ...,  2.0924e-01,
          -1.6307e-01, -2.0405e-01],
         [-9.3034e-02,  5.6803e-02, -1.4616e-01,  ...,  3.4695e-01,
          -7.9954e-02, -1.8222e-01],
         ...,
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01]],

        ...,

        [[-2.0241e-01,  4.3304e-01,  3.1586e-02,  ...,  3.5010e-01,
          -4.8267e-01, -8.3591e-02],
         [-1.1699e-01,  2.2332e-02,  6.9277e-02,  ..., -1.1184e-02,
          -1.6743e-01, -1.3263e-01],
         [ 2.1830e-01,  1.3748e-01,  1.9907e-01,  ..., -4.1769e-01,
           9.8340e-02, -1.2098e-01],
         ...,
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01]],

        [[-4.6185e-02, -5.1657e-03, -9.0559e-02,  ..., -1.5105e-01,
          -3.6440e-01, -2.5530e-02],
         [ 4.4459e-01, -6.3791e-02, -3.5578e-01,  ..., -1.3777e-02,
          -2.9397e-02,  6.2943e-02],
         [ 2.2697e-01,  7.0650e-02,  7.8798e-02,  ..., -2.5299e-01,
           2.9128e-02, -1.0424e-01],
         ...,
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01]],

        [[ 1.0809e-01,  8.7689e-02, -2.3905e-01,  ..., -4.2419e-01,
           6.5307e-02, -1.2141e-01],
         [ 7.6068e-04,  2.5588e-01,  2.2293e-01,  ..., -3.6710e-01,
          -2.3922e-02,  9.5212e-02],
         [ 2.5240e-01,  2.3380e-01,  3.9794e-02,  ..., -4.4483e-02,
          -2.6306e-01,  2.1050e-02],
         ...,
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01],
         [ 5.0378e-01,  6.4967e-01,  4.0962e-01,  ...,  6.4058e-01,
           2.7467e-01,  7.9185e-01]]], grad_fn=<EmbeddingBackward0>)
    """

    dataset_name = 'THUCNews'  # 数据集

    # args
    model_name = args.model
    embedding = args.embedding
    # args model:Text_RNN, embedding:pre_trained
    print(f"args model:{model_name}, embedding:{embedding}")

    if args.embedding == 'pre_trained':
        embedding = Embedding.sou_gou_embedding.value

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
