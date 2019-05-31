# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: C4.5决策树(连续变量）
@version: Anaconda Python 3.6
"""
from math import log, sqrt
import operator
import re
from b_DecisionTree_study.plotTree import createPlot


def createDataSet():
	dataSet = [['sunny', 85, 85, 'FALSE', 'no'],
	           ['sunny', 80, 90, 'TRUE', 'no'],
	           ['overcast', 83, 86, 'FALSE', 'yes'],
	           ['rainy', 70, 96, 'FALSE', 'yes'],
	           ['rainy', 68, 80, 'FALSE', 'yes'],
	           ['rainy', 65, 70, 'TRUE', 'no'],
	           ['overcast', 64, 65, 'TRUE', 'yes'],
	           ['sunny', 72, 95, 'FALSE', 'no'],
	           ['sunny', 69, 70, 'FALSE', 'yes'],
	           ['rainy', 75, 80, 'FALSE', 'yes'],
	           ['sunny', 75, 70, 'TRUE', 'yes'],
	           ['overcast', 72, 90, 'TRUE', 'yes'],
	           ['overcast', 81, 75, 'FALSE', 'yes'],
	           ['rainy', 71, 91, 'TRUE', 'no']]
	labels = ['outlook', 'temperature', 'humidity', 'windy']
	return dataSet, labels


def classCount(dataSet):
	"""
    计算类别个数
    :param dataSet:
    :return:
    """
	labelCount = {}
	for one in dataSet:
		if one[-1] not in labelCount.keys():
			labelCount[one[-1]] = 0
		labelCount[one[-1]] += 1
	return labelCount


def calcShannonEntropy(dataSet):
	"""
    计算数据集的香农熵
    :param dataSet:
    :return:
    """
	labelCount = classCount(dataSet)
	numEntries = len(dataSet)
	Entropy = 0.0
	for i in labelCount:
		prob = float(labelCount[i]) / numEntries
		Entropy -= prob * log(prob, 2)
	return Entropy


def majorityClass(dataSet):
	"""
    返回类别个数最多的类别
    :param dataSet:
    :return:
    """
	labelCount = classCount(dataSet)
	sortedLabelCount = sorted(labelCount.items(), reverse=True)  # 排序类别统计，倒序
	return sortedLabelCount[0][0]

def splitDataSet(dataSet, i, value):
	"""
    在数据集中选出第i个属性值为value的子集，并且去掉该属性值（删除已判断过的属性及其值）
    :param dataSet:
    :param i:
    :param value:
    :return:
    """
	subDataSet = []
	for one in dataSet:
		if one[i] == value:
			reduceData = one[:i]
			reduceData.extend(one[i + 1:])
			subDataSet.append(reduceData)
	return subDataSet

def splitContinuousDataSet(dataSet, i, value, direction):
	"""
    根据连续变量属性分割数据集
    :param dataSet:
    :param i:
    :param value:
    :param direction:
    :return:
    """
	subDataSet = []
	for one in dataSet:
		if direction == 0:
			if one[i] > value:
				reduceData = one[:i]
				reduceData.extend(one[i + 1:])
				subDataSet.append(reduceData)
		if direction == 1:
			if one[i] <= value:
				reduceData = one[:i]
				reduceData.extend(one[i + 1:])
				subDataSet.append(reduceData)
	return subDataSet


def chooseBestFeat(dataSet, labels):
	"""
    找出当前dataset最优feat及最优值
    :param dataSet:
    :param labels:
    :return:
    """
	global bestFeatValue, bestSplit
	baseEntropy = calcShannonEntropy(dataSet)  # 初始熵
	bestFeat = 0
	baseGainRatio = -1  # 初始增益率
	numFeats = len(dataSet[0]) - 1  # 特征数
	bestSplitDic = {}
	i = 0
	for i in range(numFeats):
		featVals = [example[i] for example in dataSet]  # 每种特征的特征值列表
		if type(featVals[0]).__name__ == 'float' or type(featVals[0]).__name__ == 'int':  # 如果是连续性特征
			j = 0
			sortedFeatVals = sorted(featVals)  # 特征值顺序排序
			splitList = []
			for j in range(len(featVals) - 1):
				splitList.append((sortedFeatVals[j] + sortedFeatVals[j + 1]) / 2.0)
			for j in range(len(splitList)):
				newEntropy = 0.0
				splitInfo = 0.0
				value = splitList[j]
				subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
				subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
				prob0 = float(len(subDataSet0)) / len(dataSet)
				newEntropy -= prob0 * calcShannonEntropy(subDataSet0)
				prob1 = float(len(subDataSet1)) / len(dataSet)
				newEntropy -= prob1 * calcShannonEntropy(subDataSet1)
				splitInfo -= prob0 * log(prob0, 2)
				splitInfo -= prob1 * log(prob1, 2)
				gainRatio = float(baseEntropy - newEntropy) / splitInfo
				if gainRatio > baseGainRatio:
					baseGainRatio = gainRatio
					bestSplit = j
					bestFeat = i
			bestSplitDic[labels[i]] = splitList[bestSplit]
		else: #非连续性特征
			uniqueFeatVals = set(featVals)
			splitInfo = 0.0
			newEntropy = 0.0
			for value in uniqueFeatVals:
				subDataSet = splitDataSet(dataSet, i, value)
				prob = float(len(subDataSet)) / len(dataSet)
				splitInfo -= prob * log(prob, 2)
				newEntropy -= prob * calcShannonEntropy(subDataSet)
			gainRatio = float(baseEntropy - newEntropy) / splitInfo
			if gainRatio > baseGainRatio:
				bestFeat = i
				baseGainRatio = gainRatio
	if type(dataSet[0][bestFeat]).__name__ == 'float' or type(dataSet[0][bestFeat]).__name__ == 'int': #连续特征
		bestFeatValue = bestSplitDic[labels[bestFeat]] #转为实际特征值
	if type(dataSet[0][bestFeat]).__name__ == 'str':
		bestFeatValue = labels[bestFeat]
	return bestFeat, bestFeatValue

def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if len(set(classList)) == 1:  # 如果样本全部属于同一类别,返回该类别
		return classList[0]
	if len(dataSet[0]) == 1:  # 如果数据集只有一条记录时，返回类别个数最多的类别即本身类
		return majorityClass(dataSet)
	bestFeat, bestFeatLabel = chooseBestFeat(dataSet, labels)
	myTree = {labels[bestFeat]: {}}
	subLabels = labels[:bestFeat]
	subLabels.extend(labels[bestFeat + 1:])
	if type(dataSet[0][bestFeat]).__name__ == 'str':
		featVals = [example[bestFeat] for example in dataSet]
		uniqueVals = set(featVals)
		for value in uniqueVals:
			reduceDataSet = splitDataSet(dataSet, bestFeat, value)
			myTree[labels[bestFeat]][value] = createTree(reduceDataSet, subLabels)
	if type(dataSet[0][bestFeat]).__name__ == 'int' or type(dataSet[0][bestFeat]).__name__ == 'float':
		value = bestFeatLabel
		greaterDataSet = splitContinuousDataSet(dataSet, bestFeat, value, 0)
		smallerDataSet = splitContinuousDataSet(dataSet, bestFeat, value, 1)
		myTree[labels[bestFeat]]['>' + str(value)] = createTree(greaterDataSet, subLabels)
		myTree[labels[bestFeat]]['<=' + str(value)] = createTree(smallerDataSet, subLabels)
	return myTree

def decision_tree_predict(decision_tree, attribute_labels, one_test_data):
	first_key = list(decision_tree.keys())[0]
	second_dic = decision_tree[first_key]
	attribute_index = attribute_labels.index(first_key)
	res_label = None
	for key in second_dic.keys():  # 属性分连续值和离散值，连续值对应<=和>两种情况
		if key[0] == '<':
			value = float(key[2:])
			if float(one_test_data[attribute_index]) <= value:
				if type(second_dic[key]).__name__ == 'dict':
					res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
				else:
					res_label = second_dic[key]
		elif key[0] == '>':
			value = float(key[1:])
			if float(one_test_data[attribute_index]) > value:
				if type(second_dic[key]).__name__ == 'dict':
					res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
				else:
					res_label = second_dic[key]
		else:
			if one_test_data[attribute_index] == key:
				if type(second_dic[key]).__name__ == 'dict':
					res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
				else:
					res_label = second_dic[key]
	return res_label

def getCount(inputTree, dataSet, featLabels, count):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		rightcount = 0
		wrongcount = 0
		tempfeatLabels = featLabels[:]
		subDataSet = splitDataSet(dataSet, featIndex, key)
		tempfeatLabels.remove(firstStr)
		if type(secondDict[key]).__name__ == 'dict':
			getCount(secondDict[key], subDataSet, tempfeatLabels, count)
		# 在这里加上剪枝的代码，可以实现自底向上的悲观剪枝
		else:
			for eachdata in subDataSet:
				if str(eachdata[-1]) == str(secondDict[key]):
					rightcount += 1
				else:
					wrongcount += 1
			count.append([rightcount, wrongcount, secondDict[key]])


def cutBranch_downtoup(inputTree, dataSet, featLabels, count):  # 自底向上剪枝
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():  # 走到最深的非叶子结点
		if type(secondDict[key]).__name__ == 'dict':
			tempcount = []  # 本将的记录
			rightcount = 0
			wrongcount = 0
			tempfeatLabels = featLabels[:]
			subDataSet = splitDataSet(dataSet, featIndex, key)
			tempfeatLabels.remove(firstStr)
			getCount(secondDict[key], subDataSet, tempfeatLabels, tempcount)
			# 在这里加上剪枝的代码，可以实现自底向上的悲观剪枝
			# 计算，并判断是否可以剪枝
			# 原误差率，显著因子取0.5
			tempnum = 0.0
			wrongnum = 0.0
			old = 0.0
			# 标准误差
			standwrong = 0.0
			for var in tempcount:
				tempnum += var[0] + var[1]
				wrongnum += var[1]
			old = float(wrongnum + 0.5 * len(tempcount)) / float(tempnum)
			standwrong = sqrt(tempnum * old * (1 - old))
			# 假如剪枝
			new = float(wrongnum + 0.5) / float(tempnum)
			if new <= old + standwrong and new >= old - standwrong:  # 要确定新叶子结点的类别
				'''
			#计算当前各个类别的数量多少，然后，多数类为新叶子结点的类别
		tempcount1=0
		tempcount2=0
		for var in subDataSet:
			if var[-1]=='0':
			tempcount1+=1
			else:
			tempcount2+=1
		if tempcount1>tempcount2:
			secondDict[key]='0'
		else:
			secondDict[key]='1'
				'''
				# 误判率最低的叶子节点的类为新叶子结点的类
				# 在count的每一个列表类型的元素里再加一个标记类别的元素。
				wrongtemp = 1.0
				newtype = -1
				for var in tempcount:
					if float(var[1] + 0.5) / float(var[0] + var[1]) < wrongtemp:
						wrongtemp = float(var[1] + 0.5) / float(var[0] + var[1])
						newtype = var[-1]
				secondDict[key] = str(newtype)
				tempcount = []  # 这个相当复杂，因为如果发生剪枝，才会将它置空，如果不发生剪枝，那么应该保持原来的叶子结点的结构
			for var in tempcount:
				count.append(var)
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			continue
		rightcount = 0
		wrongcount = 0
		subDataSet = splitDataSet(dataSet, featIndex, key)
		for eachdata in subDataSet:
			if str(eachdata[-1]) == str(secondDict[key]):
				rightcount += 1
			else:
				wrongcount += 1
		count.append([rightcount, wrongcount, secondDict[key]])  # 最后一个为该叶子结点的类别


def cutBranch_uptodown(inputTree, dataSet, featLabels):  # 自顶向下剪枝
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			tempfeatLabels = featLabels[:]
			subDataSet = splitDataSet(dataSet, featIndex, key)
			tempfeatLabels.remove(firstStr)
			tempcount = []
			getCount(secondDict[key], subDataSet, tempfeatLabels, tempcount)
			# 计算，并判断是否可以剪枝
			# 原误差率，显著因子取0.5
			tempnum = 0.0
			wrongnum = 0.0
			old = 0.0
			# 标准误差
			standwrong = 0.0
			for var in tempcount:
				tempnum += var[0] + var[1]
				wrongnum += var[1]
			old = float(wrongnum + 0.5 * len(tempcount)) / float(tempnum)
			standwrong = sqrt(tempnum * old * (1 - old))
			# 假如剪枝
			new = float(wrongnum + 0.5) / float(tempnum)
			if new <= old + standwrong and new >= old - standwrong:  # 要确定新叶子结点的类别
				'''
				#计算当前各个类别的数量多少，然后，多数类为新叶子结点的类别
			tempcount1=0
			tempcount2=0
			for var in subDataSet:
				if var[-1]=='0':
				tempcount1+=1
				else:
				tempcount2+=1
			if tempcount1>tempcount2:
				secondDict[key]='0'
			else:
				secondDict[key]='1'
					'''
				# 误判率最低的叶子节点的类为新叶子结点的类
				# 在count的每一个列表类型的元素里再加一个标记类别的元素。
				wrongtemp = 1.0
				newtype = -1
				for var in tempcount:
					if float(var[1] + 0.5) / float(var[0] + var[1]) < wrongtemp:
						wrongtemp = float(var[1] + 0.5) / float(var[0] + var[1])
						newtype = var[-1]
				secondDict[key] = str(newtype)

if __name__ == '__main__':
	dataSet, labels = createDataSet()
	Tree=createTree(dataSet,labels)
	# createPlot(Tree)
	# print(decision_tree_predict(Tree, labels, ['rainy', 71, 91, 'TRUE']))