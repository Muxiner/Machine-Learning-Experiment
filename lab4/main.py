import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """
    读取数据
    """
    data = np.loadtxt('./perceptron_data.txt')
    for i in data:
        i[2] = -1 if i[2] == 0 else 1
    return data[:, 0:-1], data[:, -1]


def get_scatter_data(x_axis, y_axis, label):
    """
    对数据按照 label【0，1】进行分组，方便绘制不同颜色的散点图
    """
    x_red, x_blue, y_red, y_blue = [], [], [], []
    for index in range(0, len(label)):
        if label[index] == 1:
            x_red.append(x_axis[index])
            y_red.append(y_axis[index])
        elif label[index] == -1:
            x_blue.append(x_axis[index])
            y_blue.append(y_axis[index])
    return [x_red, x_blue], [y_red, y_blue]


def draw_table(x_axis, y_axis, label):
    """
    显示最开始的分组结果
    """
    axis_x, axis_y = get_scatter_data(x_axis, y_axis, label)
    color = ['red', 'blue']
    mark = ['1', '0']
    for index in range(len(axis_x)):
        plt.scatter(axis_x[index], axis_y[index], color=color[index], label=mark[index], alpha=.5)
    plt.xlim((-4, 4))
    plt.ylim((-3, 15))
    plt.legend()  # 显示图例
    plt.show()


def table_line(x_axis, y_axis, label, omega, theta, i):
    """
    绘制每次迭代时的超平面，并保存图片
    :param x_axis: 样本的第一列
    :param y_axis: 样本第二列
    :param label: 样本标签【0，1】
    :param omega: w
    :param theta: θ
    :param i: 迭代次数
    :return:
    """
    axis_x, axis_y = get_scatter_data(x_axis, y_axis, label)
    color = ['red', 'blue']
    mark = ['1', '0']
    x1 = -theta / omega[0]
    x2 = -theta / omega[1]

    if x1 != x2:
        for index in range(len(axis_x)):
            plt.scatter(axis_x[index], axis_y[index], color=color[index], label=mark[index], alpha=.5)
        plt.xlim((-4, 4))
        plt.ylim((-3, 15))
        plt.axline([x1, 0], [0, x2], label='超平面', color='black')
        plt.legend()  # 显示图例
        plt.savefig('./result/images' + str(i) + '.jpg')
        if i < 5:
            plt.show()
        plt.close()


def get_gif():
    """
    将每次迭代的结果绘图集合成一个 GIF
    """
    import os
    import imageio
    filenames = []
    path = './result'
    for files in os.listdir(path):
        if files.endswith('jpg'):
            file = os.path.join(path, files)
            filenames.append(file)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('result.gif', images, duration=0.5)


def perceptron(data_x, data_y, eta):
    """
    感知机学习算法
    :param data_x: 样本
    :param data_y: 标签
    :param eta: 学习率
    :return: w， θ
    """
    omega = np.zeros(data_x.shape[1])
    theta = 0
    classify_count = 0
    classify_round = 0
    classify_right = False
    new_round = True
    while not classify_right:
        if new_round:
            print("第 %d 轮：" % classify_round)
            classify_round += 1
        classify_right = True
        new_round = False
        for index in range(0, data_x.shape[0]):
            if data_y[index] * (np.dot(omega, data_x[index]) + theta) <= 0:
                theta += eta * data_y[index]
                omega += eta * data_x[index] * data_y[index]
                classify_count += 1
                print("分类次数：%d\t θ: %.2f\t w：" % (classify_count, theta), omega)
                table_line(data_x[:, 0], data_x[:, 1], data_y, omega, theta, classify_count)
                classify_right = False
                new_round = True
    return omega, theta


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    xx, yy = load_data()
    draw_table(xx[:, 0], xx[:, 1], yy)
    w, b = perceptron(xx, yy, 0.5)
    get_gif()


