import numpy as np


class LDAModel(object):
    """
    线性判别分析
    """

    def __init__(self, data, target, d) -> None:
        """
        数据特征 数据标签 降维后的维度
        """
        self.data = data
        self.target = target
        self.d = d
        self.labels = set(target)
        # 列的平均值 μ
        # 所有样本的均值向量
        self.mu = self.data.mean(0)
        self.new_data = None
        self.swt_sb = None
        self.w = None
        self.Sw = None
        self.Sb = None
        self.St = None
        self.classify, self.class_mu = None, None

    def data_divide_to_vectors(self):
        """
        功能：将传入的数据集按 target 分成不同的类别集合并求出对应集合的均值向量
        """
        self.classify, self.class_mu = {}, {}
        # print(self.target, self.labels)
        # 根据结果的值【0，1】，将训练样本分类：【0】一类，【1】一类
        # 同时求得每类中各列的平均值
        for label in self.labels:
            # 分类
            self.classify[label] = self.data[self.target == label]
            # 列的平均值
            self.class_mu[label] = self.classify[label].mean(0)

    def get_sw(self):
        """
        功能：定义类内散度矩阵
        """
        self.get_st()
        self.get_sb()
        # St = Sw + Sb
        self.Sw = self.St - self.Sb

    def get_st(self):
        """
        功能：计算全局散度矩阵
        """
        self.St = np.dot((self.data - self.mu).T, (self.data - self.mu))

    def get_sb(self):
        """
        功能：计算类间散度矩阵
        """
        # 创建新的数组，
        self.Sb = np.zeros((self.data.shape[1], self.data.shape[1]))
        for i in self.labels:
            # 获取类别i样例的集合
            class_i = self.classify[i]
            # 获取类别i的均值向量
            mu_i = self.class_mu[i]
            self.Sb += len(class_i) * np.dot((mu_i - self.mu).values.reshape(-1, 1),
                                             (mu_i - self.mu).values.reshape(1, -1))

    def get_swt_sb(self):
        """
        计算 Sw^(-1)*Sb
        """
        self.data_divide_to_vectors()
        self.get_sw()
        self.swt_sb = np.linalg.inv(self.Sw).dot(self.Sb)

    def get_w(self):
        """
        功能：计算w
        """
        self.get_swt_sb()
        #  特征值 和 特征向量
        # eig_vectors[:i]与 eig_values相对应
        eig_values, eig_vectors = np.linalg.eig(self.swt_sb)
        # 寻找 d 个最大非零广义特征值
        top_d = (np.argsort(eig_values)[::-1])[:self.d]
        # 用d个最大非零广义特征值组成的向量组成w
        self.w = eig_vectors[:, top_d]

    def get_new_sample(self):
        """
        计算新的样本——已降维
        """
        self.get_w()
        self.new_data = np.dot(self.data, self.w)

    def get_result(self):
        """
        执行计算
        """
        self.get_new_sample()
