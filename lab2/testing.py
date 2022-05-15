import matplotlib.pyplot as plt

from LDA import LDAModel
from main import load_data
import numpy as np
from scipy.stats import norm


def judge_classification(gauss_dist, x):
    """
    功能：判断样本x属于哪个类别
    """
    # 将样本带入各个类别的高斯分布概率密度函数进行计算
    judge_result = [[k, norm.pdf(x, loc=v['loc'], scale=v['scale'])] for k, v in gauss_dist.items()]

    # 寻找计算结果最大的类别
    judge_result.sort(key=lambda s: s[1], reverse=True)
    # print(judge_result)
    return judge_result[0][0]


def accuracy_score(y_true, y_pred):
    """
    Compare y_true to y_pred and return the accuracy
    比较 实际值 与 预测值，并返回准确度
    """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def testing():
    """
    功能：对测试集进行分类并返回准确率
    """
    # 加载数据集
    train_x, train_y, test_x, test_y = load_data('./data/blood_data.txt')
    # print(train_x, test_x, train_y, test_y)
    # 创建模型
    lda = LDAModel(train_x, train_y, 1)
    # 获取投影矩阵w
    lda.get_result()
    # 对训练集进行降维
    train_x_handled = lda.new_data
    # 降维后的样本集
    # data_df = pd.concat([pd.DataFrame(train_x_handled), train_y], axis=1)
    # data_df.columns = ['Handled-Z1', 'Donate-Y1']
    # print(data_df)

    # 获取训练集各个类别对应的高斯分布的均值和方差
    gauss_dist = {}
    for i in lda.labels:
        category = train_x_handled[train_y == i]
        # 计算均值
        loc = category.mean()
        # 计算方差
        scale = category.std()
        gauss_dist[i] = {'loc': loc, 'scale': scale}
    # 测试集降维
    test_x_handled = np.dot(test_x, lda.w)
    pred_y = np.array([judge_classification(gauss_dist, x) for x in test_x_handled])

    # print(train_x_handled)
    # print(train_y.values)
    # # 绘图
    # print(train_x_handled[train_y == 0])
    # print(np.zeros(train_x_handled[train_y == 0].shape[1], 1))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.scatter(train_x_handled[train_y == 0], np.ones((1, train_x_handled[train_y == 0].shape[0])),
                marker='.', color='red', label='2007年3月位献血: 0', alpha=0.5)
    plt.scatter(train_x_handled[train_y == 1], np.zeros((1, train_x_handled[train_y == 1].shape[0])),
                marker='^', color='blue', label='2007年3月献血: 1', alpha=0.5)
    plt.legend()
    plt.title("训练集降维情况")
    plt.ylabel("是否献血/(1/0)")
    plt.xlabel("样本降维后的参数")
    plt.show()

    plt.scatter(test_x_handled[test_y == 0], np.ones((1, test_x_handled[test_y == 0].shape[0])),
                marker='.', color='red', label='2007年3月未献血: 0', alpha=0.5)
    plt.scatter(test_x_handled[test_y == 1], np.zeros((1, test_x_handled[test_y == 1].shape[0])),
                marker='^', color='blue', label='2007年3月献血: 1', alpha=0.5)
    plt.legend()
    plt.title("测试集降维情况")
    plt.ylabel("是否献血/(1/0)")
    plt.xlabel("样本降维后的参数")
    plt.show()

    y1 = test_y
    y2 = pred_y
    x = range(0, 148)
    plt.bar(x, y1, color='#ff4eff', label='test', alpha=0.5)
    plt.bar(x, -y2, color='#ba2c01', label='pred', alpha=0.5)
    plt.legend()
    plt.title("真实值与预测值的比较（预测的准确率：" + str(accuracy_score(test_y, pred_y)) + ")")
    plt.ylabel("是否献血/(1/0)")
    plt.xlabel("序号")
    plt.show()
    mylog = open('output.txt', mode='a+', encoding='utf-8')
    print("训练结果(w)：", lda.w.T, file=mylog)
    print("预测准确度：", accuracy_score(test_y, pred_y), file=mylog)
    mylog.close()
    # return accuracy_score(test_y, pred_y)


if __name__ == "__main__":
    print("训练1次。")
    testing()
    print("训练完成。")
