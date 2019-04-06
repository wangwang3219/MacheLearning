import numpy as np
import os
import operator

def img2vector(filename):
    returnVect = np.zeros([1, 1024])
    fr = open(filename)
    for i in range(32):
        listfr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(listfr[j])
    return returnVect

def classify(inX, trainingData, labels, k):
    lenTraining = len(trainingData)
    diffMat = np.tile(inX, (lenTraining, 1))
    diff = (diffMat - trainingData) ** 2
    diff = diff.sum(axis=1) ** 0.5
    distance = diff.argsort()                       #返回距离由大到小的索引值
    dic = {}
    for i in range(k):
        label = labels[distance[i]]
        dic[label] = dic.get(label, 0) + 1
    return sorted(dic.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def handWritting():
    hwTrainingLabels = []
    trainingName = os.listdir('trainingDigits')
    lenTraining = len(trainingName)
    diffMat = np.zeros([lenTraining, 1024])
    for i in range(lenTraining):
        hwTrainingLabels.append(int(trainingName[i].split('_')[0]))
        diffMat[i, :] = img2vector('trainingDigits/%s' % trainingName[i])
    testingName = os.listdir('testDigits')
    lenTestingName = len(testingName)
    testMat = np.zeros([1, 1024])
    error = 0.0
    for i in range(lenTestingName):
        testLabel = int(testingName[i].split('_')[0])
        testMat = img2vector('testDigits/%s' % testingName[i])
        test = classify(testMat, diffMat, hwTrainingLabels, 3)
        print('预测值为：%d, 真实值为：%d' % (test, testLabel))
        if(test != testLabel):
            error += 1.0
    print('总共错了%d个数据，错误率为：%f%%' % (error, error/lenTestingName*100))


if __name__ == '__main__':
    handWritting()