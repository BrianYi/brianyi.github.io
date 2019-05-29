import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def loadDataSet_iris():
    """数据集生成

    Returns:
        array -- 数据集
        array -- 标签集
    """
    dataMat, labelMat = load_iris(return_X_y=True)
    dataMat, labelMat = dataMat[:100, :3], labelMat[:100]
    dataMat[:, 2] = dataMat[:, 1]
    dataMat[:, 1] = dataMat[:, 0]
    dataMat[:, 0] = 1
    return dataMat, labelMat


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def hypo(theta, x):
    """假设函数(hypothesis function)

    Arguments:
        theta {array} -- 参数值
        x {array} -- 一个数据或数据集

    Returns:
        array -- 通过假设函数得到的预测值
    """
    return sigmoid(np.dot(x, theta))


def jcost(theta, X, y):
    """J代价函数

    Arguments:
        theta {array} -- 参数值
        X {array} -- 数据集
        y {array} -- 标签集

    Returns:
        array -- 代价
    """
    hX = hypo(theta, X)
    # 元素级乘法获取第一项的值(数组)
    first = np.multiply(y, np.log(hX))
    # 元素级乘法获取第二项的值(数组)
    second = np.multiply(np.ones(len(y))-y, np.log(np.ones(len(hX))-hX))
    return -(first+second)/len(X)


def partial_jcost(theta, X, y, x):
    """J代价函数关于theta的偏导数

    Arguments:
        theta {array} -- 参数值
        X {array} -- 数据集
        y {array} -- 标签集
        x {array} -- 一个数据或数据集

    Returns:
        array -- 偏导值
    """
    return np.dot(hypo(theta, X)-y, x)


def decision_boundary(prob):
    """决策边界:
           概率>=0.5时,为1
           概率<0.5时,为0

    Arguments:
        prob {array} -- 一组概率值

    Returns:
        array -- 一组类别值
    """
    return prob > 0.5*np.ones(len(prob))


def accuracy_rate(X, y, theta):
    """计算预测准确率

    Arguments:
        X {array} -- 预测数据集
        y {array} -- 已知观测值
        theta {array} -- 参数值

    Returns:
        float -- 返回预测准确率
    """
    y_predict = classify(hypo(theta, X))
    trueCount = np.sum(y_predict == y)
    return float(trueCount)/len(y)


def batch_gradient_desc(X, y, alpha=0.01, numIterations=500):
    """BGD下降法(批量梯度下降法)

    Arguments:
        X {array} -- 数据集
        y {array} -- 标签集

    Keyword Arguments:
        alpha {float} -- 步长(学习率) (default: {0.01})
        numIterations {int} -- 迭代次数 (default: {500})

    Returns:
        array -- 返回参数值
    """
    # 获取特征数n
    n = len(X[0])
    theta = np.ones(n)
    # 批量梯度下降法
    for i in range(numIterations):
        theta = theta-alpha*partial_jcost(theta, X, y, X)
    return theta


def stochastic_gradient_desc(X, y, alpha=0.01, numIterations=100):
    """SGD下降法(随机梯度下降法)

    Arguments:
        X {array} -- 数据集
        y {array} -- 标签集

    Keyword Arguments:
        alpha {float} -- 步长(学习率) (default: {0.01})
        numIterations {int} -- 迭代次数 (default: {100})

    Returns:
        array -- 返回参数值
    """
    # 获取样本数m,特征数n
    m, n = len(X), len(X[0])
    theta = np.ones(n)
    # 随机梯度下降法
    for k in range(numIterations):
        for i in range(m):
            for j in range(n):
                theta[j] = theta[j]-alpha * \
                    partial_jcost(theta, X[i], y[i], X[i, j])
    return theta


def classify(prob):
    """由概率返回分类类别

    Arguments:
        prob {array} -- 每个样本的概率

    Returns:
        array -- 返回分类类别
    """
    return decision_boundary(prob)


def auto_norm(X):
    """特征归一化(或特征缩放)

    Arguments:
        X {array} -- 数据集

    Returns:
        array -- 返回归一化后的数据集
    """
    X = np.array(X)
    n = len(X[0])
    minVals = X.min(0)
    maxVals = X.max(0)
    newVals = (X-minVals)/(maxVals-minVals)
    return newVals


def plotBestFit(theta, dataMat, labelMat, title='Gradient Descent', subplt=111):
    """绘制图像

    Arguments:
        theta {array} -- 参数列表
        dataMat {array} -- 样本集
        labelMat {array} -- 标签集

    Keyword Arguments:
        title {str} -- 图像标题 (default: {'Gradient Descent'})
        subplt {int} -- 子图 (default: {111})
    """
    # 存储分类,所有标签值0的为一类,标签值1的为一类
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    # 分类
    for i in range(len(dataMat)):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord0.append(dataMat[i][1])
            ycord0.append(dataMat[i][2])
    plt.subplot(subplt)
    plt.scatter(xcord0, ycord0, s=30, c='red', marker='s', label='failed')
    plt.scatter(xcord1, ycord1, s=30, c='green', label='success')
    x0_min = dataMat[:, 1].min()
    x0_max = dataMat[:, 1].max()
    x = np.arange(x0_min, x0_max, 0.1)
    y = (-theta[0]-theta[1]*x)/theta[2]
    plt.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()


if __name__ == "__main__":
    # 读取数据集,标签集
    dataMat, labelMat = loadDataSet_iris()
    # 特征归一化(特征缩放)
    dataMat[:,-2:]=auto_norm(dataMat[:,-2:])
    # 交叉验证:产生100个随机数做下标,将数据打乱
    rndidx = np.arange(100)
    np.random.shuffle(rndidx)
    shuffledX = []
    shuffledy = []
    for i in range(100):
        shuffledX.append(dataMat[rndidx[i]])
        shuffledy.append(labelMat[rndidx[i]])
    dataMat, labelMat = np.array(shuffledX), np.array(shuffledy)
    X, y = np.array(dataMat), np.array(labelMat)
    # 获取前50个数据做训练数据,用于训练模型
    Xtrain, ytrain = np.array(dataMat[:50]), np.array(labelMat[:50])
    # 获取后50个数据做测试数据,用于测试预测准确率
    Xtest, ytest = np.array(dataMat[-50:]), np.array(labelMat[-50:])
    # 画布大小
    plt.figure(figsize=(13, 6))
    # 使用BGD批量梯度下降法
    theta = batch_gradient_desc(Xtrain, ytrain)
    # 拿测试数据放入模型,计算预测准确率
    bgd_acc = accuracy_rate(Xtest, ytest, theta)
    # 绘图
    plotBestFit(theta, X, y, 'BGD(Batch Gradient Descent) accuracy %.2f' %
                bgd_acc, 121)
    # 使用SGD随机梯度下降法
    theta = stochastic_gradient_desc(Xtrain, ytrain)
    # 拿测试数据放入模型,计算预测准确率
    sgd_acc = accuracy_rate(Xtest, ytest, theta)
    # 绘图
    plotBestFit(theta, X, y, 'SCD(Stochastic Gradient Descent) accuracy %.2f' %
                sgd_acc, 122)
    plt.show()
