
# 分类入门——感知机(Perceptron)

用莺尾花数据集的花瓣长度和花瓣宽度这两个特征，150个样本集来学习：


```python
from sklearn import datasets
import numpy as np
```


```python
iris = datasets.load_iris()
X = iris.data[:,[2,3]] #选泽后两个特征
y = iris.target
print('Class label:',np.unique(y))#查看label种类
X.shape
```

    Class label: [0 1 2]
    




    (150, 2)



np.unique(y)返回三种莺尾花的种类：Iris-setosa, Irisversicolor,
and Iris-virginica，分别对应0 1 2。将str类型的标签用整数来代替是机器学习中的常用做法，能提高计算性能，避免一些小问题。

为了评估模型训练的好坏，需要将样本集划分为训练集和测试集。


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
```


```python
X_train.shape
```




    (105, 2)



利用**train_test_split**函数，我们**随机**将X，y数据集的30%划分为测试集(test data)，70%划分为训练集(training data)。random_state参数设置了随机种子(=1)，打乱(shuffle)了数据分布，不然原始数据是按顺序排列的，前50个是0，然后50个是1，然后50个是2。

最后一个参数stratufy=y使得**按层划分数据**，即训练集和测试集中的label是按比例划分的：


```python
print('Label counts in y:',np.bincount(y))
```

    Label counts in y: [50 50 50]
    


```python
print('Label counts in y_train:',np.bincount(y_train))
```

    Label counts in y_train: [35 35 35]
    


```python
print('Label counts in y_test:',np.bincount(y_test))
```

    Label counts in y_test: [15 15 15]
    

np.bincount函数对数组中的每种元素分别计数。由上可知，label的确被按比例划分了。

然后进行**特征缩放**(feature scaling)：


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

我们从preprocessing模块导入了StandardScaler类，并初始化了对象sc，用它调用类的方法fit、transform。注意我们用相同的参数对训练集和测试集进行缩放，以使它们互相搭配。

对数据进行标准化之后，现在可以训练感知机模型了。大多数算法在sklearn中支持多分类问题通过One-versus-Rest(OvR)方法：


```python
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std,y_train)
```




    Perceptron(alpha=0.0001, class_weight=None, eta0=0.1, fit_intercept=True,
          max_iter=40, n_iter=None, n_jobs=1, penalty=None, random_state=1,
          shuffle=True, tol=None, verbose=0, warm_start=False)



max_iter，最大迭代次数，定义了训练次数epochs，eta0即学习率learning rate。如果学习率设置过大，可能会跨过全局最低，如果学习率太小，就需要进行很多次epoch才能收敛，对于大的数据集会慢很多。因此，合适的学习率要通过多次实验判断得到。

random_state参数则保证了每一轮epoch迭代完之后能重新打乱测试集。

以上，我们已经训练完了模型，可以用它来进行预测了：


```python
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
```

    Misclassified samples: 3
    

测试集45组数据，错分类了3个，正确率为：


```python
print('正确率：%.2f%s' % (42/45*100,'%'))
```

    正确率：93.33%
    

除了自己计算正确率，sklearn也给我们提供了算法正确率的计算方法accuracy_score：


```python
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_pred))
```

    Accuracy: 0.93
    

另外，sklearn中的每个算法自带了准确率计算方法score，它集成了predict和accuracy_score两种方法：


```python
print('Accuracy: %.2f' % ppn.score(X_test_std,y_test))
```

    Accuracy: 0.93
    

最后，我们可以定义plot_decision_regions函数用来画出分类的边界：


```python
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
%matplotlib inline

def plot_decision_regions(X, y, classifier, test_idx=None,
resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8, c=colors[idx],
            marker=markers[idx], label=cl,
            edgecolor='black')
    # highlight test samples
    if test_idx:
    # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
        c='', edgecolor='black', alpha=1.0,
        linewidth=1, marker='o',
        s=100, label='test set')
```

然后用此函数画出iris数据集的决策边界：


```python
fig =plt.figure(dpi=500)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train,y_test)) #adarray的聚合，后面学习
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn,test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
```




    <matplotlib.legend.Legend at 0x25aa7839cf8>




![png](output_32_1.png)


简单的感知机只能进行**线性分类**，如果数据集线性不可分，我们就需要利用更为强大的算法来进行分类。

# 概率模型——逻辑回归(Logistic Regression)

## 概率模型

尽管感知机模型给我们提供了很方便的分类算法入门，但是它有一个致命的缺陷：**无法对非线性可分的数据集完美分类**。

接下来学习更为强大的分类器--逻辑回归。要知道，**虽然名为回归，它却是名副其实的分类算法**，可用于线性分类和二元分类问题，还可以通过OvR方法扩展到多分类问题。逻辑回归在工业应用中是最广为运用的分类算法之一。

为解释逻辑回归作为**概率模型**的原理，首先引入**odds ratio**：表示发生某件事的可能性。可以写作p/(1-p)，其中p代表要预测的结果的概率。比如要预测病人是否有某种疾病，如果有则定义y=1。同时可以定义log函数：![00124.jpeg](attachment:00124.jpeg)

log表示自然对数，它可以将0-1范围的输入转换为整个实数范围，**我们用线性关系来表示特征值和log概率之间的关系**：

![00125.jpeg](attachment:00125.jpeg)

其中，p(y=1|x)表示在给定特征x的情况下y=1的**条件概率**。

那么反过来，如果我们要预测某个特定样本属于某个类别的话，用logit的反转形式，也称之为**sigmoid function**：

![00127.jpeg](attachment:00127.jpeg)

logit函数可以将0-1范围的概率扩展到实数范围内，而sigmoid函数则可以将实数范围内的数缩放至0-1范围内。在这里，z是权重和样本特征的线性组合：

![00128.jpeg](attachment:00128.jpeg)

接下来我们画出sigmoid函数曲线来看它的样子：


```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

fig =plt.figure(dpi=500)
z = np.arange(-7,7,0.1)
phi_z = sigmoid(z)
plt.plot(z,phi_z)
plt.xlabel('z')
plt.ylabel('$\phi(z)$')
plt.yticks([0.0,0.5,1.0])
ax = plt.gca()
ax.yaxis.grid(True)
```


![png](output_47_0.png)


根据函数曲线可知，当z趋向于正无穷时函数趋近于1，z趋向于负无穷时函数趋近于0。sigmoid函数将实数范围内的数限制在了(0,1)之间。将之应用在概率分析层面：

![00142.jpeg](attachment:00142.jpeg)

二分类问题中，sigmoid函数**将输入变为概率输出**，我们认为如果概率大于0.5则输出1，不然输出0。

## 损失函数

线性回归的损失函数为sum-squared-error:

![00144.jpeg](attachment:00144.jpeg)

让损失函数最小化从而更新权重w来得到线性回归模型。

为引入逻辑回归的损失函数，我们假设各样本之间相互独立，定义我们想让之最大的似然函数L：

![00145.jpeg](attachment:00145.jpeg)

**<知识点：>**

极大似然估计：假设有一堆数据点，我们想找一个模型能够很好地描述这些数据，可以根据大致分布用一直的分布模型来套用，然后得到似然函数。而**极大似然估计的目的就是找到参数，能使模型最大化生成这些数据点的概率**。没错，就是需要求导，对于似然函数求导一般采用**对数求导**。

对上式的似然函数L取对数，有：

![00146.jpeg](attachment:00146.jpeg)

log函数的好处是，如果概率非常小的情况下不用log将难以计算，另外，可以将乘法变成加法，更容易计算。

在机器学习中我们要最小化损失函数，所以**取负的似然函数求其最小**，即为逻辑回归的损失函数：

![00147.jpeg](attachment:00147.jpeg)

为更好地理解损失函数，可以来看**单个样本**的损失函数：

![00148.jpeg](attachment:00148.jpeg)

如果y=1，损失函数为前半式，我们会希望z-->正无穷，phi_z趋向于1，log(phi_z)趋向于0，则损失函数为0。y与phi_z趋向一致。

![00150.jpeg](attachment:00150.jpeg)

接下来我们画图说明phi_z与损失函数的关系：


```python
def cost_1(z):
    return - np.log(sigmoid(z))
def cost_0(z):
    return - np.log(1-sigmoid(z))
fig =plt.figure(dpi=500)
z = np.arange(-10,10,0.1)
phi_z = sigmoid(z)
c1 = [cost_1(x) for x in z]
plt.plot(phi_z,c1,label='J(w) if y=1')
c0 = [cost_0(x) for x in z]
plt.plot(phi_z,c0,linestyle='--',label='J(w) if y=0')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
```




    <matplotlib.legend.Legend at 0x25aa96a0b00>




![png](output_69_1.png)


## 用逻辑回归训练模型

数据集依然是莺尾花，使用sklearn中的**linear_model.LogisticRegression**模型


```python
from sklearn.linear_model import LogisticRegression

fig = plt.figure(dpi=500)
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,classifier=lr,test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
```




    <matplotlib.legend.Legend at 0x25aa7839978>




![png](output_72_1.png)


样本属于某个类别的概率可以用**predict_proba**方法预测，前三个测试样本属于三个分类的概率如下：


```python
lr.predict_proba(X_test_std[:3,:])
```




    array([[  3.20136878e-08,   1.46953648e-01,   8.53046320e-01],
           [  8.34428069e-01,   1.65571931e-01,   4.57896429e-12],
           [  8.49182775e-01,   1.50817225e-01,   4.65678779e-13]])




```python
lr.predict_proba(X_test_std[:3, :]).sum(axis=1)
```




    array([ 1.,  1.,  1.])



每行的相加和为1，每一行为样本分别属于三种类别的概率。第一行最大值为0.853，意味着第一个样本属于类别三(Iris-virginica)的概率为0.853。即判断第一个样本属于类别三。我们可以用argmax函数得到最大值出现的索引：


```python
lr.predict_proba(X_test_std[:3, :]).argmax(axis=1) #axis=1 表示按行，与pandas中正好相反。
```




    array([2, 0, 0], dtype=int64)



当然，以上步骤得到的预测结果在sklearn也有函数可以直接得到，即**predict**：


```python
lr.predict(X_test_std[:3, :])
```




    array([2, 0, 0])



最后，如果想要预测**单个样本**的话：sklearn要求predict的输入是**二维数组**，我们必须将一维数组变为二维数组才能进行预测，此处需要**reshape**方法：


```python
lr.predict(X_test_std[0, :].reshape(1, -1)) #注意是reshape（1，-1），
```




    array([2])



## 正则化降低过拟合

**过拟合（Overfitting）**是机器学习中常见的问题，模型在训练集上拟合过度导致不能泛化的测试集。如果一个模型过拟合，我们说它具有**高方差(high variance)**。同样的，模型也有可能**欠拟合(Underfitting)**，这意味着我们的模型甚至不能拟合训练集，更不用说去泛化测试集，它具有**高偏差(high bias)**。

![00165.jpeg](attachment:00165.jpeg)

让方差、偏差达到平衡的一种方法是正则(regularization)。正则可以很好地过滤到数据中的噪点来防止过拟合。它的思想是增加新的信息（bias）来惩罚极端的参数(weight)。常见的正则形式有L2正则：

![00166.jpeg](attachment:00166.jpeg)

lambda 称为正则参数。

正则是**特征缩放(feature scaling)**的另一个重要原因。要使得正则能够完美生效，我们需要保证所有的**特征(features)在同一数量级**。

增加正则之后的逻辑回归损失函数如下：

![00168.jpeg](attachment:00168.jpeg)

通过正则参数lambda，我们可以控制（使权重weights小时的）训练集拟合度。**增大lambda，就增强了正则强度，增大了惩罚力度**。

LogisticRegression参数中的C来自于支持向量机SVM，**C与lambda负相关**，lambda增大则C减小。后续SVM中会讨论到。

减小C值就相当于加大正则的惩罚力度，我们来画图说明L2正则对莺尾花数据集中的两个特征，花瓣长度、花瓣宽度的权重的影响：


```python
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)  
weights = np.array(weights)

fig =plt.figure(dpi=500)
plt.plot(params, weights[:, 0],label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
```


![png](output_94_0.png)


上图可以看到，C值在0.01以下时，对损失函数惩罚力度大，**正则项很大，要使整个损失函数最小，前半式只能为0**。C值增大，惩罚力度减小，得到w值。

# 最大间隔——支持向量机

另一个强大且广为应用的机器学习算法——支持向量机（Support Vector Machine）,可以看作是感知机的扩展。感知机中我们要最小化误分类的个数，在SVM中我们的优化目标是最大化间隔(margin)。间隔指的是划分超平面与最近的样本间的距离，最近的样本也被称为**支持向量support vectors**。

![00170.jpeg](attachment:00170.jpeg)

## 硬间隔

假设样本**线性可分**，也被称为**硬间隔**。让决策边界与样本间具有大的间隔，可以降低泛化误差。相反，小的间隔就可能会引起过拟合。定义**正负样本平面方程**：

![00171.jpeg](attachment:00171.jpeg)
![00172.jpeg](attachment:00172.jpeg)

两式相减，得到：

![00173.jpeg](attachment:00173.jpeg)

然后用向量**w**的模来对上式标准化，**w**的模：

![00174.jpeg](attachment:00174.jpeg)

得到下面的式子：

![00175.jpeg](attachment:00175.jpeg)

点到平面的距离公式：

![%E7%82%B9%E5%88%B0%E5%B9%B3%E9%9D%A2%E7%9A%84%E8%B7%9D%E7%A6%BB%E5%85%AC%E5%BC%8F.png](attachment:%E7%82%B9%E5%88%B0%E5%B9%B3%E9%9D%A2%E7%9A%84%E8%B7%9D%E7%A6%BB%E5%85%AC%E5%BC%8F.png)

由上面的分析可知，点到平面的距离公式为**向量与法向量的点积除以法向量的模**。

由此可知，上面推导出的式子左边就是距离决策平面最近的**正负样本间的距离**，就是我们想要最大化的间隔（margin）。

现在，求解SVM最大间隔问题就变成了求解![00176.jpeg](attachment:00176.jpeg)最大值的问题了。通常会写作**最小化**![00181.jpeg](attachment:00181.jpeg)

在此情况下，正负样本的约束条件就变成了：

![00177.jpeg](attachment:00177.jpeg)
![00178.jpeg](attachment:00178.jpeg)
![00179.jpeg](attachment:00179.jpeg)

其中，N为样本总数。

上面两个方程说明，所有的正样本都应该在正平面一侧，负样本在负平面一侧，也可以写作下式：

![00180.jpeg](attachment:00180.jpeg)

于是得到我们要求的结果和其约束条件：

![%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA.png](attachment:%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA.png)

s.t.:subject to 表示约束条件，即为了使所有的样本都在正负平面以外。要求解这个问题需要使用**拉格朗日乘子法**，它是用来求解在约束条件下极值的，以后再详细学习~

## 软间隔—引入松弛变量

以上的推导形式都是建立在**样本数据线性可分**的基础上，如果样本数据你中有我我中有你（**线性不可分**），应该如何处理呢？这里就需要引入**软间隔**（Soft Margin），意味着，允许支持向量机在**一定程度上出错**。

![%E8%BD%AF%E9%97%B4%E9%9A%94.png](attachment:%E8%BD%AF%E9%97%B4%E9%9A%94.png)

为了使目标函数可以在一定程度不满足约束条件，我们引入常数C和0/1损失函数

![01%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png](attachment:01%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)

当z<0时函数为1，否则函数为0

![%E8%BD%AF%E9%97%B4%E9%9A%94%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png](attachment:%E8%BD%AF%E9%97%B4%E9%9A%94%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)

对于上式来说，C>=0 是个常数，**当C无穷大时，迫使所有样本均满足约束；当C取有限值时，允许一些样本不满足约束**：

![00186.jpeg](attachment:00186.jpeg)

但是上式的0/1损失函数非凸、非连续，数学性质不好，我们用其他一些函数来代替它：

![%E6%9B%BF%E4%BB%A3%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png](attachment:%E6%9B%BF%E4%BB%A3%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)

以上4中损失函数图像如下所示：

![%E6%9B%BF%E4%BB%A3%E5%87%BD%E6%95%B0%E5%9B%BE%E5%83%8F.png](attachment:%E6%9B%BF%E4%BB%A3%E5%87%BD%E6%95%B0%E5%9B%BE%E5%83%8F.png)

为了书写方便，我们引入**松弛变量(slack variable)**，可将目标函数重写为：

![%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0png.png](attachment:%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0png.png)

上式就是常见的**软间隔支持向量机**，其中，每一个样本都有一个对应的松弛变量，用以表征该样本不满足约束的程度。

其中参数C与正则相关，增大C就增加了对目标函数的惩罚。

## 模型训练

学习了支持向量机的原理，接下来我们用SVM模型来训练莺尾花数据集：


```python
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std,y_train)

fig =plt.figure(dpi=500)
plot_decision_regions(X_combined_std,y_combined,
                      classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
```




    <matplotlib.legend.Legend at 0x25aaab3cef0>




![png](output_140_1.png)


对比逻辑回归和支持向量机，它们通常会得到相似的结果。逻辑回归试图最大化训练集分布的可能性，它对于异常值要比支持向量机更敏感，而支持向量机更关注的是靠近决策边界的那部分支持向量。另一方面，逻辑回归的优势在于它运行很简单，运行速度也比SVM要快，还有，它更容易更新，在处理流数据时更有利。

以上我们利用sklearn实现了了感知机(Perceptron)、逻辑回归(LogisticRegression)和支持向量机(Support Vector Machine)对莺尾花数据集的预测。如果数据集很大的情况下，用原来的方法进行预测可能会很慢，我们可以进行类似**随机梯度下降法的运行方式，SGDClassifier**：


```python
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
```

## 非线性决策边界--核函数(Kernel)
