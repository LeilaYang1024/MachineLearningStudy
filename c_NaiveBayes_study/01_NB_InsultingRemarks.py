# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: 基于朴素贝叶斯模型（伯努利）的侮辱性言语过滤
@version: Anaconda Python 3.6
"""
import numpy as np

def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec

def createVocabList(dataset):
	"""创建文档的词汇表"""
	vocabSet=set([])
	for docoument in dataset:
		vocabSet=vocabSet | set(docoument)
	return list(vocabSet)

def setOfWords2Vec(VocabList,input):
	"""创建词集模型(只判断词是否出现）"""
	returnVec=[0]*len(VocabList)
	for word in input:
		if word in VocabList:
			returnVec[VocabList.index(word)]=1
		else:
			print("{0}不在词汇表中".format(word))
	return returnVec

def TrainNB(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix)
	numWords=len(trainMatrix[0])
	pAbusive=sum(trainCategory)/float(numTrainDocs) #1的概率
	p0Num=np.ones(numWords) #初始化分子为1，分母为2
	p1Num=np.ones(numWords)
	p0Denom=2.0
	p1Denom=2.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1Num+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
		else:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	p1Vect=np.log(p1Num/p1Denom) #防止过多小数相乘造成过拟合
	p0Vect=np.log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vect,p1vect,pClass1):
	p1=sum(vec2Classify*p1vect)+np.log(pClass1)
	p0=sum(vec2Classify*p0Vect)+np.log(1.0-pClass1)
	if p1>p0:
		return 1
	else:
		return 0
	
def testingNB():
	postingList, classVec = loadDataSet()
	VocabList = createVocabList(postingList)
	trainMat=[]
	for postDoc in postingList:
		trainMat.append(setOfWords2Vec(VocabList,postDoc))
	p0Vect, p1Vect, pAbusive=TrainNB(trainMat,classVec)
	testEntry = ['stupid', 'garbage']
	thisDoc = np.array(setOfWords2Vec(VocabList, testEntry))
	print("测试文本：{0}".format(testEntry))
	print("测试结果：{0}".format(classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)))



if __name__=="__main__":
	testingNB()
	