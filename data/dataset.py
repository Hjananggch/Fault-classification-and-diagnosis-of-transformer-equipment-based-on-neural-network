import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):

    def __init__(self, dir, if_normalize=False):
        super(dataset, self).__init__()

        if (dir.endswith('.csv')):
            data = pd.read_csv(dir)
        else:
            return

        nplist = data.T.to_numpy()
        data = nplist[0:-1].T
        label = nplist[-1]
        self.data = np.float64(data)

        # Tensor化
        self.data = torch.FloatTensor(self.data)
        self.label = torch.LongTensor(label)

        #归一化
        if if_normalize == True:
            self.data = nn.functional.normalize(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

def data_split(data, rate):
    train_l = int(len(data) * rate)
    test_l = len(data) - train_l
    """打乱数据集并且划分"""
    train_set, test_set = torch.utils.data.random_split(data, [train_l, test_l])
    return train_set, test_set

if __name__ == '__main__':
    # 示例用法
    csv_file_path = 'a.csv'  # 请替换成你的CSV文件路径
    result_data = dataset(csv_file_path,if_normalize=True)
    print(result_data[100])
