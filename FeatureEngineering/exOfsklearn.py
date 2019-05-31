# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: sklearn 提供特征处理方法
@version: Anaconda Python 3.6
"""
import numpy as np

#1.导入iris数据集
from sklearn.datasets import load_iris
iris=load_iris()
#特征矩阵
dataset=iris.data
#目标向量
targetvec=iris.target

#2.数据预处理(使用sklearn中的preproccessing库来进行数据预处理)
#2.1.1无量纲化--标准化
from sklearn.preprocessing import StandardScaler
StandData=StandardScaler().fit_transform(dataset)
#2.1.2无量纲化--区间缩放[0,1]
from sklearn.preprocessing import MinMaxScaler
MMData=MinMaxScaler().fit_transform(dataset)
#2.1.3无量纲化--归一化
from sklearn.preprocessing import Normalizer
NMData=Normalizer().fit_transform(dataset)
#2.2对定量特征二值化(设定阈值：threshold)
from sklearn.preprocessing import Binarizer
BData=Binarizer(threshold=3).fit_transform(dataset)
#2.3对定性特征哑编码
dataset2=iris.target.reshape((-1,1))
from sklearn.preprocessing import OneHotEncoder
OHEData=OneHotEncoder().fit_transform(dataset2).toarray()
#2.4缺失值计算
newdataset=np.vstack((dataset,np.array(['nan','nan','nan','nan']))) #给数据集加入缺失
from sklearn.preprocessing import Imputer
IData=Imputer().fit_transform(newdataset)
#2.5 数据变换(不理解)

#3.特征选择(使用sklearn中的feature_selection库来进行特征选择)
#3.1Filter(过滤法)
#3.1.1方差选择法(threshold:设置方差阈值)
from sklearn.feature_selection import VarianceThreshold
VTData=VarianceThreshold(threshold=0).fit_transform(dataset)
#3.1.2相关系数法
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
def multivariate_pearsonr(X, y):
	"""数据集的每个特征矩阵与结果矩阵计算皮尔逊相关系数,返回二元组（评分，P值）"""
	scores, pvalues = [],[]
	for ret in map(lambda x:pearsonr(x,y),X.T):
		scores.append(abs(ret[0]))
		pvalues.append(ret[1])
	return (np.array(scores), np.array(pvalues))
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
Xt_pearson = SelectKBest(score_func=multivariate_pearsonr,k=2).fit_transform(dataset,targetvec)
#3.1.3卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
Xt_chi2=SelectKBest(score_func=chi2,k=2).fit_transform(dataset,targetvec)
#3.1.4互信息法
from sklearn.feature_selection import SelectKBest
from minepy import MINE
def mic(x, y):
	"""由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5  """
	m = MINE()
	m.compute_score(x, y)
	return (m.mic(), 0.5)
def multivariate_mi(X, y):
	"""数据集的每个特征矩阵与结果矩阵计算最大信息系数,返回二元组（评分，P值固定0.5）"""
	scores, pvalues = [], []
	for ret in map(lambda x:mic(x,y),X.T):
		scores.append(ret[0])
		pvalues.append(ret[1])
	return (np.array(scores), np.array(pvalues))
Xt_mi=SelectKBest(score_func=multivariate_mi, k=2).fit_transform(dataset,targetvec)
#3.2Wrapper(包装法)
#3.2.1递归特征消除法
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
Xt_rfe=RFE(estimator=LogisticRegression(),n_features_to_select=2).fit_transform(dataset,targetvec)
#3.3Embedded(嵌入法)
#3.3.1基于惩罚项的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
Xt_sfm=SelectFromModel(LogisticRegression(penalty="l1",C=0.1)).fit_transform(dataset,targetvec)


#4.降维
def plot_A(Xt,targetvec):  #降维效果绘图
	red_x, red_y = [], []
	blue_x, blue_y = [], []
	green_x, green_y = [], []
	for i in range(len(Xt_pca)):
		if targetvec[i] == 0:
			red_x.append(Xt_pca[i][0])
			red_y.append(Xt_pca[i][1])
		elif targetvec[i] == 1:
			blue_x.append(Xt_pca[i][0])
			blue_y.append(Xt_pca[i][1])
		else:
			green_x.append(Xt_pca[i][0])
			green_y.append(Xt_pca[i][1])
	plt.scatter(red_x, red_y, c='r', marker='x')
	plt.scatter(blue_x, blue_y, c='b', marker='D')
	plt.scatter(green_x, green_y, c='g', marker='.')
	plt.show()
	
#4.1主成分分析法(PCA)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
Xt_pca=PCA(n_components=2).fit_transform(dataset)
print(Xt_pca)
plot_A(Xt_pca,targetvec)
#4.2线性判别分析法(LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Xt_lda=LinearDiscriminantAnalysis(n_components=2).fit_transform(dataset,targetvec)
print(Xt_lda)
plot_A(Xt_lda,targetvec)

