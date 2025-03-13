import torch

class DatasetIterator(object):

    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.index = 0
        self.num_batches = len(dataset) // batch_size  # batch数量
        self.residue = len(self.dataset) % self.num_batches != 0  # batch数量是否正好为整数

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):  # 迭代器
        if self.residue and self.index == self.num_batches:
            # 取最后非batch_size大小段
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.num_batches:
            self.index = 0
            raise StopIteration
        else:
            # 取batch_size下一段
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):  # 可迭代对象
        return self

    def __len__(self):  # 容器对象
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches

def build_iterator(dataset, batch_size, device):
    return DatasetIterator(dataset, batch_size, device)
