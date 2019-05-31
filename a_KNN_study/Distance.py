# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: k-近邻算法的几种距离计算：
		1.空间：欧氏距离
		2.路径：曼哈顿距离
		3.国际象棋国王：切比雪夫距离
		4.“欧式+曼哈顿+切比雪夫”的统一形式:闵可夫斯基距离
		5.加权：标准化欧氏距离
		6.排除量纲和依存：马氏距离
		7.向量差距：夹角余弦
		8.编码差别：汉明距离
		9.集合近似度：杰卡德类似系数与距离
		10.相关：相关系数与相关距离
@version: Anaconda Python 3.6
"""
from numpy import *

def EuclideanDistance(dataSet,input):
	"""计算欧式距离"""
	Dist=sqrt((square(dataSet-input)).sum(1))
	return  Dist

def ManhattanDistance(dataSet,input):
	"""计算曼哈顿距离"""
	Dist=(abs(dataSet-input)).sum(1)
	return Dist
	
def ChebyshevDistance(dataSet,input):
	"""计算切比雪夫距离"""
	Dist=(abs(input - dataSet)).max(1)
	return Dist

def StandardizedEuclideanDistance(dataSet,input):
	"""计算标准化欧式距离"""
	sk=var(dataSet,axis=0,ddof=1)
	Dist=sqrt((square(dataSet - input)/sk).sum(1))
	return Dist


if __name__=="__main__":
	setA = array([[0.1, 0.5, 0.2], [0.2, 0.3, 0.9]])
	setB = array([[0.5, 0.5, 0.5]])
	# print(euclideanDistance(setA,setB))
	# print(manhattanDistance(setA,setB))
	# print(chebyshevDistance(setA,setB))
	# print(StandardizedEuclideanDistance(setA,setB))
	