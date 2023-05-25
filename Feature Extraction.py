import numpy as np
import pandas as pd


def stripColumn(columns: np.ndarray) -> np.ndarray:
    result = []
    for idx in range(len(columns)):
        result.append(str(columns[idx]).strip())
    return np.array(result)


def createPearson():
    except_col_list = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]

    pd_csv = pd.read_csv("./dataset/re_sort_DDos_data.csv", index_col=0)
    pd_csv.columns = stripColumn(pd_csv.columns)
    pd_csv = pd_csv.drop(except_col_list, axis=1)  # 상관 분석에서 제외할 CSV COLUMNS
    # pd_csv['Label'] = pd_csv['Label'].apply(labelMapping)
    pearson = pd_csv.corr(method='pearson')['Label']
    pearson.to_csv("./pearson_result.csv")


pearson = pd.read_csv("./pearson_result.csv")

for i in [0.5, 0.4, 0.3, 0.2]:
    pear = pearson[(pearson['Label'] >= i) | (pearson['Label'] <= -1 * i)]
    print(i, pear, len(pear), '\n')
