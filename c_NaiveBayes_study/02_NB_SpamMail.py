# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: 基于朴素贝叶斯的垃圾邮件过滤模型
@version: Anaconda Python 3.6
"""

import re
import random
import numpy as np

def bagOfWords2Vec(VocabList,input):
	"""创建词袋模型（包含词的出现次数）"""
	returnVec=0*len(VocabList)
	for word in input:
		returnVec[VocabList.index(word)]+=1
	return returnVec

def setOfWords2Vec(VocabList,input):
	"""创建词集模型(只判断词是否出现）"""
	returnVec=[0]*len(VocabList)
	for word in input:
		if word in VocabList:
			returnVec[VocabList.index(word)]=1
		else:
			print("{0}不在词汇表中".format(word))
	return returnVec

def createVocabList(dataset):
	"""创建文档的词汇表"""
	vocabSet=set([])
	for docoument in dataset:
		vocabSet=vocabSet | set(docoument)
	return list(vocabSet)

def textParse(bigstring):
	"""解析语料并标准化"""
	listOftokens=re.split(r'\W*',bigstring)  #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
	return [tok.lower() for tok in listOftokens if len(tok) > 2]  #除了单个字母，例如大写的I，其它单词变成小写
	
def TrainNB(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix)
	numWords=len(trainMatrix[0])
	pAbusive=sum(trainCategory)/float(numTrainDocs) #1的概率
	p0Num=np.ones(numWords) #初始化分子为1，分母为2
	p1Num=np.ones(numWords)
	p0Denom=1.0
	p1Denom=1.0
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

def SpamMailTest():
	docList=[]
	classList=[]
	fullText=[]
	for i in range(1,26):
		wordList=textParse(open("../c_NaiveBayes_study/email/spam/%d.txt" % i,'r').read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(1)
		wordList = textParse(open("../c_NaiveBayes_study/email/ham/%d.txt" % i,'r').read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)
	trainingSet=list(range(50)) #训练集索引
	testSet=[] #测试集索引
	for i in range(10):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del trainingSet[randIndex]
	trainMat=[]
	trainClasses=[]
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0,p1,pA=TrainNB(np.array(trainMat),np.array(trainClasses))
	errorCount=0.0
	for docIndex in testSet:
		testVect=setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(np.array(testVect),p0,p1,pA) != classList[docIndex]:
			errorCount+=1
	print("失误率：{0}".format(errorCount/len(testSet)))
	

if __name__=="__main__":
	SpamMailTest()