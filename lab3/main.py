import math
import numpy as np


def load_data():
    """
    读取数据到数组，并分为训练集 和 测试集
    x, y 表示 特征，标签
    """
    data = np.loadtxt("./data/lenses_data.txt", dtype=int, usecols=[1, 2, 3, 4, 5])
    # print(data)
    # data_x, data_y = data[:, 0:-1], data[:, -1]
    # testing_x, testing_y = data_x[14:-1, :], data_y[14:-1]
    # return training_x, training_y, testing_x, testing_y
    return data


def equal_num(x, label):
    """
    计算 x 在 label 中的数量
    """
    num = 0
    for i in x:
        num += 1 if i == label else 0
    return num


def get_info_entropy(x):
    """
    计算信息熵
    """
    info_entropy = 0
    for i in set(x):
        p_i = equal_num(x, i) / x.size
        info_entropy -= p_i * math.log(p_i, 2)
    return info_entropy


def get_conditional_info_entropy(feature, x):
    """
    计算某特征的条件信息熵
    :param feature: 特征
    :param x: 标签
    :return:
    """
    # 特征的标签的种类
    feature_label = set(feature)
    # 标签的种类
    label = set(x)
    conditional_info_entropy = 0
    for i in feature_label:
        # 某一特征标签所占比例
        p_i = equal_num(feature, i) / feature.size
        # x[feature == i] 是指特征标签对应的标签
        # get_information_entropy(x[feature == i]) 计算特征标签对应的标签的条件信息熵
        # print(x[feature == i])
        conditional_info_entropy += p_i * get_info_entropy(x[feature == i])
    return conditional_info_entropy


def get_info_gain_ratio(feature, x):
    """
    计算某特征的信息增益率
    :param feature: 某一特征
    :param x: 标签
    :return:
    """
    # 某特征信息增益
    info_gain = get_info_entropy(x) - get_conditional_info_entropy(feature, x)
    # 某特征信息增益率
    # 要注意某特征的固有值 —— 信息熵 可能会为 0
    ratio = 0 if get_info_entropy(feature) == 0 else info_gain / get_info_entropy(feature)
    return ratio


def get_best_feature(data, label, is_print):
    """
    获取具有最佳信息增益率的特征
    :param data: 数据集
    :param label: 标签
    :param is_print: 是否打印入文件
    :return: 最佳信息增益率的特征的索引
    """
    index = 0
    best_info_gain_ratio = 0
    # 遍历每一列，计算每个特征的信息增益率
    for row in range(0, data.shape[1]):
        info_gain_ratio = get_info_gain_ratio(data[:, row], label)
        if is_print:
            print("第 %d 个特征的信息增益率为 %.17f" % (row + 1, info_gain_ratio),
                  file=out_file)
        # index = row if info_gain_ratio > best_info_gain_ratio else index
        # best_info_gain_ratio = info_gain_ratio if info_gain_ratio > best_info_gain_ratio else best_info_gain_ratio
        if info_gain_ratio > best_info_gain_ratio:
            index = row
            best_info_gain_ratio = info_gain_ratio

    # print("最佳的信息增益率： %0.17f\n所在位置：第 %d 个特征" % (best_info_gain_ratio, index))
    return index


def split_dataset(data, index, value):
    """
    对数据集进行分片
    :param data: 需要分片的数据集
    :param index: 新数据集不需要的某列的索引
    :param value: 用于确定新数据集需要某行在 某列[index]对应的值
    :return: 新的数据集
    """
    has_split_dataset = []
    for col_data in data:
        # 选择特定行的数据进行切片
        if col_data[index] == value:
            # 取 第 index 个前的所有元素
            has_split_col = col_data[:index].tolist()
            # 再加上第 index 个后面的所有元素
            has_split_col.extend(col_data[index + 1:].tolist())
            # 再将其加入 新的 数据集
            has_split_dataset.append(has_split_col)
    return np.array(has_split_dataset)


def most_label(data):
    """
    返回，出现次数最大多的标签
    """
    label = list(set(data))
    most = ''
    most_count = 0
    for item_label in label:
        count = 0
        for item_data in data:
            if item_label == item_data:
                count += 1
        most = item_label if count > most_count else most
        most_count = count
    return most


def create_decision_tree(data, feature_label):
    """
    创建决策树
    :param data: 数据集
    :param feature_label: 特征集
    :return:
    """
    labels = [item[-1] for item in data]
    # 若是标签中全是属于同一类，停止分类
    # 返回该类标签
    if 1 == len(set(labels)):
        return labels[0]

    # 特征集为空 或是 特征集上的取值相同，停止分类
    # 返回出现次数最多的 标签
    # len(data[0]) == 1 就是数据集里只剩下标签，没有特征了
    # len(set(data[:, 0])) == 1 特征集上的取值相同
    # not feature_label 特征集空了
    if len(data[0]) == 1 or len(set(data[:, 0])) == 1 or not feature_label:
        return most_label(labels)

    # 从特征标签中选择最优划分标签
    # 选择最优标签
    best_index = get_best_feature(data[:, 0:-1], np.array(labels), False)
    # 获取最优的标签
    best_feature_label = feature_label[best_index]
    # 根据最优特征的标签生成树
    decision_tree = {best_feature_label: {}}

    # 得到训练集中所有最优特征的标签
    feat_value = [item[best_index] for item in data]
    # 去掉重复值
    for value in set(feat_value):
        decision_tree[best_feature_label][value] = create_decision_tree(
            split_dataset(data, best_index, value), split_feature(feature_label, best_index))
    return decision_tree


def split_feature(label, index):
    error_deal = []
    for i in range(0, len(label)):
        if i != index:
            error_deal.append(label[i])
    return error_deal


if __name__ == '__main__':
    feature_labels = ['年龄', '症状', '散光', '眼泪数量']
    dataset = load_data()
    out_file = open('output.txt', mode='a', encoding='utf-8')
    get_best_feature(dataset[:, 0:-1], dataset[:, -1], True)
    print(create_decision_tree(dataset, feature_labels), file=out_file)
    out_file.close()

