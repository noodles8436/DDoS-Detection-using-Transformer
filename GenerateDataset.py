import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Read CSV
df = pd.read_csv("./dataset/re_sort_DDos_data.csv", index_col=0, encoding='utf8')
df = df.reset_index(drop=True)

# Get Features
selected_features = ["Protocol", "Fwd Pkt Len Min", "Fwd Pkt Len Std",
                     "Bwd Pkt Len Min", "Pkt Len Min", "PSH Flag Cnt",
                     "ACK Flag Cnt", "CWE Flag Count", "Fwd Seg Size Min"]

# Preprocessing
for feat in tqdm(selected_features):
    max = df[feat].max()
    min = df[feat].min()
    df[feat] = df[feat].apply(lambda x: (x - min) / (max - min))



# Create Dataset
TimeStepSize = 3
for idx in tqdm(range(TimeStepSize-1, len(df.index), TimeStepSize)):
    # TimeStep 저장할 List 선언
    sample = []

    # target값
    y = 0

    # TimeStep 추출하여 List에 저장
    for _timestep in range(TimeStepSize):

        # 더이상 추출할 데이터셋이 없으면 종료
        if idx+_timestep >= len(df.index):
            exit(0)

        sample.append(df.loc[idx+_timestep, selected_features])
        if int(df.loc[idx+_timestep, 'Label']) == 1:
            y = 1

    # 시간 순서대로 뒤집기
    sample.reverse()

    # numpy array로 변환
    npz_sample = np.array(sample)

    # 저장하기
    np.savez(os.path.join("./dataset_preprocessed", str(idx)), x=npz_sample, y=y)