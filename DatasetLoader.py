import os
import numpy as np
from keras.utils import Sequence
from tqdm import tqdm
import random

DATASET_ROOT = "./dataset_preprocessed"
dataList = os.listdir("./dataset_preprocessed")
random.seed(42)
random.shuffle(dataList)

trainList = dataList[:int(len(dataList) * 0.8)]
valList = dataList[int(len(dataList) * 0.8):int(len(dataList) * 0.9)]
testList = dataList[int(len(dataList) * 0.9):]


class Dataset(Sequence):

    def __init__(self, batch_size, trainType):
        self.batch_size = batch_size
        self.x = []
        self.y = []

        if trainType == 0:
            self.fileList = trainList
        elif trainType == 1:
            self.fileList = valList
        elif trainType == 2:
            self.fileList = testList
        else:
            raise Exception("[ Graph Dataset ] 잘못된 trainType이 전달되었습니다!")

        for _fileName in tqdm(self.fileList):
            _dataPath = os.path.join(DATASET_ROOT, _fileName)
            data = np.load(_dataPath, allow_pickle=True)
            self.x.append(np.array(data["x"]).astype(float))
            self.y.append(np.array(data["y"]).astype(int))
            
            if len(self.x) > 50000:
                break

        self.on_epoch_end()

        self.seq_num = self.x[0].shape[0]

    def on_epoch_end(self):
        self.index = np.arange(len(self.x))
        np.random.shuffle(self.index)

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        batch_index = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i in batch_index:
            batch_x.append(self.x[i])
            batch_y.append(self.y[i])

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return batch_x, batch_y
