# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: 基于K近邻的手写识别模型
@version: Anaconda Python 3.6
"""

import numpy as np
import operator
from os import listdir


def img2vector(filename):
	"""把32*32的二进制图像矩阵转换成1*1024的向量矩阵"""
	returnVect = np.zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32 * i + j] = int(lineStr[j])
	return returnVect

def KNNclassify(inX,dataset,labels,k):
	"""
	分类模型
	:param inX:输入数据
	:param dataset:数据样本
	:param labels:样本类别
	:param k:最近邻数
	:return:分类
	"""
	m=dataset.shape[0]
	#计算样本距离
	diffMat=np.tile(inX,(m,1))-dataset
	sqdiffMat=diffMat**2
	sqDistance=sqdiffMat.sum(axis=1)
	distances=sqDistance**0.5
	#距离排序
	sortedDistIndices=distances.argsort()
	#记录类别
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndices[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def handwritingClassTest():
	hwLabels=[]
	trainningFileList=listdir("../a_KNN_study/trainingDigits") #列出训练样本文件名
	m=len(trainningFileList)
	trainingMat=np.zeros((m,1024))
	for i in range(m):
		fileNameStr=trainningFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:]=img2vector("../a_KNN_study/trainingDigits/"+fileNameStr)
	testFileList=listdir("../a_KNN_study/testDigits")
	errorCount=0.0
	mtest=len(testFileList)
	for i in range(mtest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		testMat1=img2vector("../a_KNN_study/testDigits/"+fileNameStr)
		classfiResult=KNNclassify(testMat1,trainingMat,hwLabels,3)
		print("预测结果：{0}，实际结果：{1}".format(classfiResult,classNumStr))
		if classNumStr!=classfiResult:
			errorCount+=1
	errorRatio=errorCount/float(mtest)
	print("失误率：{0}".format(errorRatio))
	
if __name__ == "__main__":
	handwritingClassTest()
	
	
