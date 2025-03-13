import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def get_time_dif(start_time):
    # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def _load_dataset(path, vocab, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):  # 自动打印进度信息
            lin = line.strip()  # 去除收尾空格
            if not lin:  # 跳过空行
                continue
            content, label = lin.split('\t')
            words_line = []
            token = [y for y in content]  # 分字
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))  # 不足补 PAD
                else:
                    token = token[:pad_size]  # 超过最大长度就截断
                    seq_len = pad_size  # 重新设定序列长度
            for word in token:  # 将单词/字符转换为索引
                words_line.append(vocab.get(word, vocab.get(UNK)))   # UNK 代表未知词
            contents.append((words_line, int(label), seq_len))  # 结构：(词索引列表, 标签, 序列长度)
    return contents

def build_dataset(model, config):
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")
    train = _load_dataset(config.train_path, vocab, model.pad_size)
    dev = _load_dataset(config.dev_path, vocab, model.pad_size)
    test = _load_dataset(config.test_path, vocab, model.pad_size)
    return vocab, train, dev, test
