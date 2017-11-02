from sklearn.datasets import load_iris #导入莺尾花数据集
iris=load_iris()
print (iris.feature_names) #显示特征名
print (iris.target_names)#显示目标种类莺尾花名称
print (iris.data[0])
print (iris.target[0])
for i in range(len(iris.target)):
    print ("Example %d: lable %s, festures %s" % (i,iris.target[i],iris.data[i]))#打印出所有莺尾花数据