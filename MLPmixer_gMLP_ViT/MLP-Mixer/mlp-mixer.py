import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import torch
import numpy as np
from torch.utils.data import DataLoader


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {}  # dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            if pid not in self.index_dic:
                self.index_dic[pid] = [index]
            else:
                self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]   # 每个pid对应的索引
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])  # 复制一份
            if len(idxs) < self.num_instances:        # 如果少于num_instances,随机选择
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)  # 打乱
            batch_idxs_dict[pid] = [idxs[i*self.num_instances : (i+1)*self.num_instances] for i in range(len(idxs)//self.num_instances)]
#             batch_idxs = []
#             for idx in idxs:
#                 batch_idxs.append(idx)
#                 if len(batch_idxs) == self.num_instances:
#                     batch_idxs_dict[pid].append(batch_idxs)
#                     batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class ImageDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length


Xp_train = torch.randn((12, 300))
yp_train = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
train_data = ImageDataSet(Xp_train, yp_train)
train_loader = DataLoader(dataset=train_data, batch_size=4, sampler=RandomIdentitySampler(train_data, 4, 2))

for data, label in tqdm(train_loader):
    print(data.shape)
    print(label)
