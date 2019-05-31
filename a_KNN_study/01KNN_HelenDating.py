# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: 基于K近邻的喜好选择模型
@version: Anaconda Python 3.6
"""
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

def showdatas(datingDataMat, datingLabels):
	"""
	可视化数据
	:param datingDataMat:
	:param datingLabels:
	:return:
	"""
	#设置汉字格式
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	#将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
	#当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
	fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))
	numberOfLabels = len(datingLabels)
	LabelsColors = []
	for i in datingLabels:
		if i == 1:
			LabelsColors.append('black')
		if i == 2:
			LabelsColors.append('orange')
		if i == 3:
			LabelsColors.append('red')
	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	plt.setp(axs0_title_text, size=9, weight='bold', color='red')
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
	axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
	plt.setp(axs1_title_text, size=9, weight='bold', color='red')
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

	#画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
	axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
	plt.setp(axs2_title_text, size=9, weight='bold', color='red')
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
	#设置图例
	didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',
	                  markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',
	                  markersize=6, label='largeDoses')
	#添加图例
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
	#显示图片
	plt.show()

def file2matrix(filename):
	fr=open(filename)
	arrayOLines=fr.readlines()
	numberOfLines=len(arrayOLines)
	ReturnMat=np.zeros((numberOfLines,3))
	classLabelVector=[]
	index=0
	for line in arrayOLines:
		line=line.strip()
		listFromLine=line.split('\t')
		ReturnMat[index,:]=listFromLine[0:3]
		#分类标记
		if listFromLine[-1] == 'didntLike':
			classLabelVector.append(1)
		elif listFromLine[-1] == 'smallDoses':
			classLabelVector.append(2)
		elif listFromLine[-1] == 'largeDoses':
			classLabelVector.append(3)
		index+=1
	return ReturnMat,classLabelVector

def autoNorm(dataset):
	"""
	数据标准：归一化
	:param dataset:
	:return:
	"""
	minVals=dataset.min(0)
	maxVals=dataset.max(0)
	ranges=maxVals-minVals
	#dataset行数
	m=dataset.shape[0]
	normDataSet=dataset-np.tile(minVals,(m,1))
	normDataSet=normDataSet/np.tile(ranges,(m,1))
	return normDataSet,minVals,ranges

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
	
def datingClassTest():
	"""
	模型测试
	:return:
	"""
	datingMat, datingLabels = file2matrix("datingTestSet.txt")
	#取数据的的百分之十作为测试样本
	hoRatio=0.10
	normMat,minVals,ranges=autoNorm(datingMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0.0
	for i in range(numTestVecs):
		classfiResult=KNNclassify(normMat[i],normMat[numTestVecs:],datingLabels[numTestVecs:],4)
		print("预测结果：{0},实际结果：{1}".format(classfiResult,datingLabels[i]))
		if classfiResult!=datingLabels[i]:
			errorCount+=1
		errorRatio=errorCount/float(numTestVecs)
	print("失误率：{0}".format(errorRatio))
	
if __name__=="__main__":
	datingClassTest()
	