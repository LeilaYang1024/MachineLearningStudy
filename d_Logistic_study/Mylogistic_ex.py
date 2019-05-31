# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: 从疝气病症状预测病马的死亡率
@version: Anaconda Python 3.6
"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from numpy import *

def sigmoid(z):
	return longfloat(1.0/(1+exp(-z)))

def stocGradAscent(data,label,numIter=200):
	"""
	随机梯度上升
	:param data:
	:param label:
	:param numIter:
	:return:
	"""
	m,n=shape(data)
	weights=ones(n)
	for j in range(numIter):
		dataIndex=list(range(m))
		for i in range(m):
			alpha=4/(1.0+i+j)+0.01
			randIndex=int(random.uniform(0,len(dataIndex)))
			h=sigmoid(sum(data[randIndex]*weights))
			error=label[randIndex]-h
			weights=weights+alpha*error*data[randIndex]
			del(dataIndex[randIndex])
	return weights

def GradAscent(data,label,alpha=0.01,numIter=500):
	"""
	梯度上升算法
	:param data:
	:param label:
	:param alpha: 步长
	:param numIter: 迭代次数
	:return:
	"""
	dataMatrix=mat(data)
	labelMat=mat(label).transpose()
	m,n=shape(dataMatrix)
	weights=ones((n,1))
	for k in range(numIter):
		h=sigmoid(dataMatrix*weights)
		error=labelMat-h
		weights=weights+alpha*dataMatrix.transpose()*error
	return weights.getA()

def classifyVector(inX,weights):
	prob=sigmoid(sum(inX*weights))
	if prob > 0.5:
		return 1
	else:
		return 0

def colicTest():
	frTrain=open("horseColicTraining.txt")
	frTest=open("horseColicTest.txt")
	trainSet=[]
	trainLabels=[]
	for line in frTrain.readlines():
		currLine=line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine) - 1):
			lineArr.append(float(currLine[i]))
		trainSet.append(lineArr)
		trainLabels.append(float(currLine[-1]))
	# Trainweights=stocGradAscent(array(trainSet),trainLabels,500)
	Trainweights=GradAscent(trainSet,trainLabels,0.0001,1)
	errorcount=0
	numvector=0
	for line in frTest.readlines():
		numvector+=1
		currLine=line.strip().split('\t')
		lineArr=[]
		for i in range(len(currLine) - 1):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr),Trainweights)) != int(currLine[-1]):
			errorcount+=1
	errRate=errorcount/numvector
	print("失误率：%.2f" % (errRate))
	
	
if __name__=="__main__":
	colicTest()

	