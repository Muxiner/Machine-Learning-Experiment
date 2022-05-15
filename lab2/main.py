# 机器学习实验二：线性判别分析
# 献血预测

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(file_path, df_columns):
    data = pd.read_csv(file_path, header=None, sep=',')
    data.columns = df_columns
    return data


def get_training_data(df, number):
    # 先去前XXX行作为训练集，后将其打乱
    return df.head(number)
    # .sample(frac=1).reset_index(drop=True)


def get_testing_data(df, number):
    # 先去前XXX行作为测试集集，后将其打乱
    return df.tail(number)
    # .sample(frac=1).reset_index(drop=True)


def load_data(file_path):
    """
    加载数据集，并划分训练集和测试集
    """
    data_columns = ['R', 'F', 'M', 'T', 'Donate']
    data_df = read_data(file_path, data_columns)

    # print(data_df.isnull().any())

    # print(data_df)
    # 训练集，测试集中顺序被打乱
    training_data = get_training_data(data_df, 600)
    testing_data = get_testing_data(data_df, 148)
    # 训练集 x, y
    training_x, training_y = training_data.iloc[:, :(training_data.shape[1] - 1)], training_data.iloc[:, -1]
    # print(set(training_y))
    # 测试集 x, y
    testing_x, testing_y = testing_data.iloc[:, :(testing_data.shape[1] - 1)], testing_data.iloc[:, -1]
    return training_x, training_y, testing_x, testing_y


if __name__ == '__main__':
    path = './data/blood_data.txt'
    # print(load_data(path))
    train_x, train_y, test_x, test_y = load_data(path)
    print(train_x.dtypes)
    print(train_x.isnull().any())
    print(train_y.isnull().any())
    print(test_x.isnull().any())
    print(test_y.isnull().any())
