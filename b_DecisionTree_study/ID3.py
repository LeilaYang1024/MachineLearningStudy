# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: ID3决策树
@version: Anaconda Python 3.6
"""
from math import log
import operator
from b_DecisionTree_study.plotTree import createPlot

#创建数据集
def createDataSet():
    dataSet = [['sunny','hot','high','FALSE','no'],
               ['sunny','hot','high','TRUE','no'],
               ['overcast','hot','high','FALSE','yes'],
               ['rainy','mild','high','FALSE','yes'],
               ['rainy','cool','normal','FALSE','yes'],
               ['rainy','cool','normal','TRUE','no'],
               ['overcast','cool','normal','TRUE','yes'],
               ['sunny','mild','high','FALSE','no'],
               ['sunny','cool','normal','FALSE','yes'],
               ['rainy','mild','normal','FALSE','yes'],
               ['sunny','mild','normal','TRUE','yes'],
               ['overcast','mild','high','TRUE','yes'],
               ['overcast','hot','normal','FALSE','yes'],
               ['rainy','mild','high','TRUE','no']]
    labels = ['outlook','temperature','humidity','windy']  # 两个特征
    return dataSet, labels

#从当前许多类别中挑出数目最多的类别
def majorityClass(classList):
    list={}
    for one in classList:
        if one not in list.keys():
            list[one]=0
        list[one]+=1
    sortedClass=sorted(list.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClass[0][0]

#计算给定数据集的信息熵
def calcShannonEnt(dataSet):
    numberData=len(dataSet)
    classCount={}
    shannonEnt=0
    for one in dataSet:
        currentLabel=one[-1]
        if currentLabel not in classCount.keys():
            classCount[currentLabel]=0
        classCount[currentLabel]+=1
    for key in classCount:
        prob=float(classCount[key])/numberData
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

#在数据集中选出第axis个属性值为value的子集，并且去掉该属性值（删除已判断过的属性及其值）
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for examle in dataSet:
        if(examle[axis]==value):
            reducedDataSet=examle[:axis]
            reducedDataSet.extend(examle[axis+1:])
            retDataSet.append(reducedDataSet)
    return retDataSet

#在数据集中选出最优的划分属性，返回值为索引值
def chooseBestFeat(dataSet):
    numberFeats=len(dataSet[0])-1 #数据集特征总数
    baseEntropy=calcShannonEnt(dataSet) #初始熵
    baseInfoGain=0
    bestFeat=-1
    newEntropy=0
    for i in range(numberFeats):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        for one in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,one)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy-=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>baseInfoGain):
            bestFeat=i
            baseInfoGain=infoGain
    return bestFeat

#对数据集创建树
def CreateTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if(classList.count(classList[0])==len(classList)):
        return classList[0]
    if(len(dataSet[0])==1):
        return majorityClass(classList)
    bestFeat=chooseBestFeat(dataSet)
    bestLabel=labels[bestFeat]
    Tree={bestLabel:{}}
    del(labels[bestFeat])
    featVals=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featVals)
    for value in uniqueVals:
        subLabels=labels[:]
        #对分割后的数据子集递归创建树
        Tree[bestLabel][value]=CreateTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return Tree

if __name__ == '__main__':
    dataSet,labels=createDataSet()
    Tree=CreateTree(dataSet,labels)
    createPlot(Tree)
    