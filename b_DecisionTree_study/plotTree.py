# coding: utf-8
"""
Created on ****-**-**
@author: YangLei
@brief: 决策树可视化API(嵌套字典类型的树结构）
@version: Anaconda Python 3.6
"""
import matplotlib.pyplot as plt

def getNumLeaves(mytree):
	"""
	获取决策树的叶子节点
	:param mytree:
	:return:
	"""
	numleaves=0 #初始化叶节点数
	firstStr=list(mytree.keys())[0] #获取第一个根节点
	secondDict=mytree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict': #判断是否还是字典，如果不是字典说明是叶节点
			numleaves+=getNumLeaves(secondDict[key])
		else:
			numleaves+=1
	return numleaves
	
def getTreeDepth(mytree):
	"""
	获取决策树的深度
	:param mytree:
	:return:
	"""
	maxDepth=0 #初始化深度
	firstStr=list(mytree.keys())[0]
	secondDict=mytree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth=1+getTreeDepth(secondDict[key])
		else:
			thisDepth=1
		#取深度最大值
		if thisDepth>maxDepth:
			maxDepth=thisDepth
	return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
	"""
	绘制线上的文字
	:param cntrPt:
	:param parentPt:
	:param txtString:
	:return:
	"""
	xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算文字的x坐标
	yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]  # 计算文字的y坐标
	createPlot.ax1.text(xMid, yMid, txtString)

# 设置画节点用的盒子的样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8") #主节点样式
leafNode = dict(boxstyle="round4", fc="0.8") #叶子节点样式
# 设置画箭头的样式    http://matplotlib.org/api/patches_api.html#matplotlib.patches.FancyArrowPatch
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	# annotate函数是为绘制图上指定的数据点xy添加一个nodeTxt注释
	# nodeTxt是给数据点xy添加一个注释，xy为数据点的开始绘制的坐标,位于节点的中间位置
	# xycoords设置指定点xy的坐标类型，xytext为注释的中间点坐标，textcoords设置注释点坐标样式
	# bbox设置装注释盒子的样式,arrowprops设置箭头的样式
	'''
	figure points:表示坐标原点在图的左下角的数据点
	figure pixels:表示坐标原点在图的左下角的像素点
	figure fraction：此时取值是小数，范围是([0,1],[0,1]),在图的左下角时xy是（0,0），最右上角是(1,1)
	其他位置是按相对图的宽高的比例取最小值
	axes points : 表示坐标原点在图中坐标的左下角的数据点
	axes pixels : 表示坐标原点在图中坐标的左下角的像素点
	axes fraction : 与figure fraction类似，只不过相对于图的位置改成是相对于坐标轴的位置
	'''
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
	                        xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', \
	                        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotTree(mytree,parentPt,nodeTxt):
	"""
	绘制树
	:param mytree:
	:param parentPt:节点坐标
	:param nodeTxt:节点注释
	:return:
	"""
	numLeaves=float(getNumLeaves(mytree))
	firstStr=list(mytree.keys())[0]
	cntrPt = (plotTree.xoff + (1.0 + float(numLeaves)) / 2.0 / plotTree.totalW, plotTree.yoff)
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = mytree[firstStr]
	# 计算节点y方向上的偏移量，根据树的深度
	plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			# 递归绘制树
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			# 更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.totalW
			plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
			# 绘制非叶子节点
			plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff),cntrPt, leafNode)
			# 绘制箭头上的标志
			plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
	plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD
	
def createPlot(inTree):
	"""
	绘制决策树
	:param inTree:
	:return:
	"""
	#创建画板
	fig=plt.figure(1,facecolor='white')#白色背景板
	axprops = dict(xticks=[], yticks=[])#定义横纵坐标轴
	#创建一个子图ax1
	createPlot.ax1=plt.subplot(111, frameon=False,**axprops)#无边框，无坐标轴
	plotTree.totalW=float(getNumLeaves(inTree))
	plotTree.totalD=float(getTreeDepth(inTree))
	plotTree.xoff = -0.5 / plotTree.totalW
	plotTree.yoff = 1.0
	plotTree(inTree,(0.5,1.0), '')
	plt.show()
	
if __name__=="__main__":
	tree={'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
	createPlot(inTree=tree)