import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def loadDataSet_iris():
    """数据集生成

    Returns:
        array -- 数据集
        array -- 标签集
    """
    dataMat, labelMat = load_iris(return_X_y=True)  # 多分类(3类)
    #dataMat, labelMat = dataMat[:100],labelMat[:100] # 二分类
    n = len(dataMat[0])
    return dataMat, labelMat


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def hypo(x, theta):
    """假设函数(hypothesis function)

    Arguments:
        theta {array} -- 参数值
        x {array} -- 一个数据或数据集

    Returns:
        array -- 通过假设函数得到的预测值
    """
    return sigmoid(np.dot(x, theta))


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
    return np.dot((hypo(X, theta)-y), x)


def predict(X, hDict):
    """预测结果

    Arguments:
        X {array} -- 测试数据集
        hDict {dict} -- 存储每个类别的参数值

    Returns:
        array -- 返回预测类别结果
    """
    m = len(X)
    y_predict = []
    for i in range(m):
        maxPro = -1.0
        maxLabel = 0
        for key in hDict.keys():
            theta = hDict[key]
            pro = hypo(X[i], theta)
            if pro > maxPro:
                maxPro = pro
                maxLabel = key
        y_predict.append(maxLabel)
    return y_predict


def accuracy_rate(X, y, hDict):
    """计算预测准确率

    Arguments:
        X {array} -- 测试数据集
        y {array} -- 已知测试结果
        hDict {dict} -- 存储每个类别的参数值

    Returns:
        float -- 准确率
    """
    m, n = len(X), len(X[0])
    y_predict = predict(X, hDict)
    true_count = np.sum(y_predict == y)
    return float(true_count)/m


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
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+k+i)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            theta=theta-alpha*np.multiply(hypo(X[randIndex], theta)-y[randIndex],X[randIndex])
    return theta


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


def plotBestFit(dataMat, labelMat, title='Gradient Descent', subplt=111):
    """绘制图像

    Arguments:
        theta {array} -- 参数列表
        dataMat {array} -- 样本集
        labelMat {array} -- 标签集

    Keyword Arguments:
        title {str} -- 图像标题 (default: {'Gradient Descent'})
        subplt {int} -- 子图 (default: {111})
    """
    dataDict = {}
    # 分类绘图
    for i in range(len(dataMat)):
        if labelMat[i] not in dataDict:
            dataDict[labelMat[i]]=[]
        dataDict[labelMat[i]].append(dataMat[i])
    plt.subplot(subplt)
    for y in dataDict.keys():
        X=np.array(dataDict[y])
        plt.scatter(X[:,1], X[:,2], s=30, label=y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()


def train(X, y):
    """训练模型,获取每个类别的参数值
    
    Arguments:
        X {array} -- 输入训练数据集
        y {array} -- 输入训练标签集
    
    Returns:
        dict -- 返回每个类别的参数值
    """
    dataDict = {}
    m = len(y)
    for i in range(m):
        if y[i] not in dataDict.keys():
            dataDict[y[i]] = []
        dataDict[y[i]].append(X[i])
    hDict = {}

    for key in dataDict.keys():
        dataX = []
        dataY = []
        for k, v in dataDict.items():
            dataX.extend(v)
            if k == key:
                dataY.extend(np.ones(len(v)))
            else:
                dataY.extend(np.zeros(len(v)))
        # 这里采用的是批量梯度下降,用随机梯度下降也是可行的
        theta = batch_gradient_desc(dataX, dataY)
        hDict[key] = theta
    return hDict


if __name__ == "__main__":
    # 读取数据集,标签集
    dataMat, labelMat = loadDataSet_iris()
    m = len(dataMat)
    # 特征归一化(特征缩放)
    dataMat[:, :] = auto_norm(dataMat[:, :])
    # 所有数据的特征增加一列x0为1
    dataMat = np.column_stack((np.ones(m), dataMat))
    # 交叉验证:将数据打乱
    rndidx = np.arange(m)
    np.random.shuffle(rndidx)
    shuffledX = []
    shuffledy = []
    for i in range(m):
        shuffledX.append(dataMat[rndidx[i]])
        shuffledy.append(labelMat[rndidx[i]])
    dataMat, labelMat = np.array(shuffledX), np.array(shuffledy)
    X, y = np.array(dataMat), np.array(labelMat)
    mTrain = int(0.75*m)
    mTest = m-mTrain
    # 获取前mTrain个数据做训练数据,用于训练模型
    Xtrain, ytrain = np.array(dataMat[:mTrain]), np.array(labelMat[:mTrain])
    # 获取后mTest个数据做测试数据,用于测试预测准确率
    Xtest, ytest = np.array(dataMat[-mTest:]), np.array(labelMat[-mTest:])
    # 画布大小
    plt.figure(figsize=(13, 6))

    # 显示原始数据
    plotBestFit(X, y, 'Original Data', 131)

    # 自己实现的LogisticRegression预测
    hDict = train(Xtrain, ytrain)
    y_predict = predict(Xtest, hDict)
    sgd_acc = accuracy_rate(Xtest, ytest, hDict)
    plotBestFit(Xtest, y_predict, 'Prediction Data (accuracy %.2f)' %
                sgd_acc, 132)

    # sklearn LogisticRegression预测
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)
    clf.fit(Xtrain[:, 1:], ytrain)   # sklearn不需要前面的x0,直接放入特征即可
    y_predict = clf.predict(Xtest[:, 1:])
    sgd_acc = np.sum(y_predict == ytest)/len(ytest) # 直接用clf.score()也可以
    plotBestFit(Xtest, y_predict,
                'Sklearn Prediction Data (accuracy {0:.2f})'.format(sgd_acc), 133)

    plt.show()
