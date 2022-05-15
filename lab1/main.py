import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# 数据集下载地址：https://www.kaggle.com/datasets/altavish/boston-housing-dataset?resource=download


def get_boston_house_data():
    """
    从包中加载波士顿房价数据集
    （本不知道有这种包，但是发现下载的数据集在操作上都存在一些问题，因而找到这种方法）
    :return:
    """
    # 加载波士顿房价数据集信息
    boston = datasets.load_boston()
    # 将房价信息转成dataFrame格式
    df_boston_house_data = pd.DataFrame(boston.data, columns=boston.feature_names)
    df_boston_house_data['PRICE'] = boston.target
    # 查看数据是否存在空值，从结果来看数据不存在空值。
    # print(df_boston_house_data.isnull().sum())
    # 查看数据大小
    # print(df_boston_house_data.shape)
    # 显示数据前5行
    # print(df_boston_house_data.head())
    # 将数据集写入csv文件
    df_boston_house_data.to_csv("boston_housing_data.csv", index=False, sep=',')
    # return df_boston_house_data


def get_data_describe(df):
    """

    :param df:
    :return:
    """
    # 查看数据的描述信息，在描述信息里可以看到每个特征的均值，最大值，最小值等信息
    df.describe()


def read_data(file_path):
    """
    从文件路径中读取数据集
    :param file_path: 文件路径
    :return: 返回一个 dataFrame
    """
    return pd.read_csv(file_path, sep=',')


def get_training_data(df, num):
    """
    从 df 中获得训练集
    :param num:
    :param df:
    :return:
    """
    # 取测试集的前450项
    df_training = df.head(num)
    # print(df_training)
    return df_training


def get_test_data(df, num):
    """
    从 df 中获取测试集
    :param df:
    :param num:
    :return:
    """
    # 取数据集的后50项
    df_test = df.tail(num)
    # print(df_test)
    return df_test


def get_last_column(df):
    """
    获取df最后一列的数据
    :param df:
    :return:
    """
    # print(df.iloc[:, -1])
    return df.iloc[:, -1]


def get_not_last_column(df):
    """
    获取df的除最后一列的所有数据
    :param df:
    :return:
    """
    # print(len(df))
    # print(df.iloc[:, :(df.shape[1] - 1)])
    return df.iloc[:, :(df.shape[1] - 1)]


def dataframe_to_matrix(df):
    """
    将df转换为多维矩阵
    其实就是转为 numpy 数组
    :param df: 需要转换的 df
    :return:
    """
    return df.to_numpy()


def model_calculation(df_x, df_y):
    """
    计算模型
    :param df_x: 训练集的前13列
    :param df_y: 训练集的第14列
    :return:
    """
    # 定义大小为 450 的全1数组，用于在 X 矩阵前插入一列全 1 的数
    one_matrix = np.full(len(df_x), 1, dtype=int)
    # X 第一列插入 一列 1，获得 X 矩阵
    x_matrix = np.insert(dataframe_to_matrix(df_x), 0, values=one_matrix, axis=1)
    # y 矩阵
    y_matrix = dataframe_to_matrix(df_y)
    # X，y 通过计算获得 β 参数
    beta = np.dot(np.linalg.inv(np.dot(x_matrix.T, x_matrix)), np.dot(x_matrix.T, y_matrix))
    # print(beta)
    return beta


def get_estimate(beta, df_test):
    """

    :param beta: β的矩阵
    :param df_test: 测试集
    :return: 返回预测价格的矩阵（数组）
    """
    # 先定义预测值的数组，大小等于测试样例大小，用0填充
    estimate = np.full(len(df_test), 0, dtype=int)
    # 定义大小为56的全1数组，用于在 X 矩阵前插入一列全 1 的数
    one_matrix = np.full(len(df_test), 1, dtype=int)
    # 将测试集中的前13项作为 矩阵 X 的数
    test_matrix = dataframe_to_matrix(get_not_last_column(df_test))
    # X 第一列插入 一列 1
    test_matrix = np.insert(test_matrix, 0, values=one_matrix, axis=1)
    # print(np.size(test_matrix, 0))
    # X 中每一行同 β进行计算，获得预测的价格
    for i in range(np.size(test_matrix, 0)):
        estimate[i] = np.dot(beta, test_matrix[i])

    # print(estimate)
    return estimate


if __name__ == '__main__':
    path = 'boston_housing_data.csv'
    # 读取csv文件，存入 df
    df_housing_data = read_data(path)
    # 获取训练集
    df_training_data = get_training_data(df_housing_data, 450)
    # 获取测试集
    df_test_data = get_test_data(df_housing_data, 56)
    # 获取训练来的 参数
    beta_matrix = model_calculation(get_not_last_column(df_training_data), get_last_column(df_training_data))
    # 测试集与训练结果参数计算获得 预测价格
    estimate_value_price = get_estimate(beta_matrix, df_test_data)
    # 实际价格
    true_value_price = dataframe_to_matrix(get_last_column(df_test_data))
    # 绘制预测值与真实值图
    # 规定字体，避免乱码
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(true_value_price, color="r", label="实际价格")  # 颜色表示
    plt.plot(estimate_value_price, color=(0, 0, 0), label="预测价格")
    plt.xlabel("测试序号")  # x轴命名表示
    plt.ylabel("价格")  # y轴命名表示
    plt.title("实际值与预测值折线图")
    plt.legend()  # 增加图例
    plt.show()  # 显示图片
