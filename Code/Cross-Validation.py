# coding: utf-8
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score



df = pd.read_csv("C:\\Users\\cyt\\Desktop\\before1035.csv")

feature_names = ['Indoor_Temperature', 'Globe_temperature',
                 'Humidity', 'Mean_radiant_temperature',
                 'Wind_Speed','PMV','Set_Point','Outdoor_T']

x = np.array(df[feature_names].values)
y = np.array(df['W'].values)
x=x.astype('int')
y=y.astype('int')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

## Logistic Regression
# estimator = LogisticRegression()
# estimator_name = "LogisticRegression"

# Decision Tree
estimator = GradientBoostingClassifier()
estimator_name = "GradientBoostingClassifier"

### Support Vector Machine
# estimator = SVC(kernel='rbf', C=8.172448979591836,gamma=0.710204081632653)
# estimator_name = "SVC"

from sklearn.model_selection import learning_curve

from matplotlib.pyplot import MultipleLocator

# It is good to randomize the data before drawing Learning Curves
def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = Y[permutation]
    return X2, Y2

a=0
# def draw_learning_curves(X, y, estimator, estimator_name):
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 303))
#     global a
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#     a=a+1
#     print(a)
#     plt.title("Learning Curves")
#     plt.ylabel("Score")
#     print('ok1')
#     plt.plot(train_scores_mean, 'o-', color="r",linewidth='1.0',
#              label="Training score")
#     plt.plot(test_scores_mean, 'o-', color="blue",linewidth='1.0',
#              label="test score")
#     y = MultipleLocator(0.05)
#     # 设置刻度间隔
#     ax = plt.gca()
#     ax.yaxis.set_major_locator(y)
#     plt.ylim(0, 1)
#     plt.legend(loc="best")
#     plt.show()
# # num_trainings = len(y)/10
# # print(len(y),num_trainings)
# X2, y2 = randomize(x, y)
#
# draw_learning_curves( x, y,estimator,GradientBoostingClassifier)

# from sklearn.decomposition import PCA
# # 降维，将四维特征转化为2维点
# X = PCA(2).fit_transform(x)
# X=x
# y=y.astype('int')
# plt.scatter(X[:,0],X[:,1],c=y)
#
# kernel = ['linear','rbf','poly','sigmoid']
# fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,4))
# for i,core in enumerate(kernel,1):
#
#     xx,yy = ((min(X[:,0]-1),max(X[:,0]+1)),(min(X[:,1]-1),max(X[:,1]+1)))
#     xx = np.arange(xx[0],xx[1],step=0.1)
#     yy = np.arange(yy[0],yy[1],step=0.1)
#     XX,YY = np.meshgrid(xx,yy)
#     grid = np.c_[XX.ravel(),YY.ravel()]
#
#     # 预测类别
#     model = SVC(kernel=core,gamma="scale",decision_function_shape="ovo",degree=3,C=1.0)
# #     model = SVC(kernel=core,gamma="scale",decision_function_shape="ovr",degree=3,C=1.0)
#     model.fit(X,y)
#     score=model.score(X,y)
#     prediction = model.predict(grid).reshape(XX.shape)
#     if i-1==0:
#         ax[0].set_title("raw data",fontsize=20)
#         ax[0].scatter(X[:,0],X[:,1],c=y,s=40)
#     ax[i].set_title(core,fontsize=20)
#     ax[i].scatter(X[:,0],X[:,1],c=y,label="%.2f"% score,s=40,zorder=10)
#     ax[i].scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=70,edgecolor="red",facecolor="none",zorder=10)
#     ax[i].pcolormesh(XX,YY,prediction,cmap=plt.cm.Accent,shading="auto")
#     ax[i].contour(XX,YY,prediction,colors=["pink","blue","pink"],linestyles=['dashed','solid','dashed'])
#     ax[i].legend(loc=4)
# plt.show()
# from sklearn.datasets import make_circles
#
#
# # 绘制3维图像需要引入mplot3d模块
# # 动态调整上下、左右翻转角度
#
# X,y = make_circles(100,noise=0.1,factor=0.5,random_state=100)
# plt.scatter(X[:,0],X[:,1],c=y)
# # 下图原始数据图
# def plot_svc(model,canvas=None):
# # 创建画布
#     if canvas is None:
#         canvas = plt.gca()
#     xlim = canvas.get_xlim()
#     ylim = canvas.get_ylim()
#     # 创建网格
#     axisx = np.linspace(xlim[0],xlim[1],num=30)
#     axisy = np.linspace(ylim[0],ylim[1],num=30)
#     axisx,axisy = np.meshgrid(axisx,axisy)
#     grid = np.stack([axisx.ravel(),axisy.ravel()])
#     # 获取距离
#     distance = model.decision_function(grid.T).reshape(axisx.shape)
#     # 绘制支持向量
#     canvas.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=70,edgecolor="red",facecolor="none",zorder=10)
#     # 绘制分界轮廓
#     canvas.contour(axisx,axisy,distance,levels=[-1,0,1],linestyles=['dashed','solid','dashed'],alpha=0.6)
#
#
# model = SVC(kernel="rbf",C=8.172448979591836,gamma=0.710204081632653).fit(X,y)
# plt.scatter(X[:,0],X[:,1],c=y)
# plot_svc(model)
# # 创建z轴刻度
# r = np.exp(-(X**2).sum(1))
# rlim = np.linspace(min(r),max(r),num=15)
# # interact装饰器动态调整上下、左右角度
#
# def plot_3d(elev=30,azim=30):
#     figure = plt.subplot(projection="3d")
#     figure.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.BrBG,edgecolors="orange")
#     figure.view_init(elev=elev,azim=azim)
#     figure.set_xlabel("x")
#     figure.set_ylabel("y")
#     figure.set_zlabel("z")
# plot_3d()
#
# plt.show()
# score = []
# C_range = np.linspace(0.01,100,50)
# for i in C_range:
#     clf = SVC(kernel="rbf",C=i,gamma = 'auto',cache_size=5000).fit(x_train,y_train)
#     score.append(clf.score(x_test,y_test))
# print(max(score), C_range[score.index(max(score))])
# plt.plot(C_range,score,color='blue', linewidth='1.0', label='C range score')
# plt.ylabel('C range score')
# plt.show()
#
# ###   grid search start
# score = []
# B=np.linspace(0.1,30,50)
# for i in B :
#     clf = SVC(kernel="rbf",C=8.172448979591836,gamma = i,cache_size=5000).fit(x_train,y_train)
#     score.append(clf.score(x_test,y_test))
#
# print(max(score), B[score.index(max(score))])
# plt.plot(B, score,color='red', linewidth='1.0', label='Gamma range score')
# plt.ylabel('Gamma range score')
# plt.show()
# ####   grid search end

# test_score = svm.score(x_test,y_test)
# print("Best score:{:.2f}".format(best_score))
# print("Best parameters:{}".format(best_parameters))
# print("Score on testing set:{:.2f}".format(test_score))

# def select_c_function(i):
# # 将含有参数的模型实例化
#     svm_model = SVC(kernel='rbf',C=i)
# # 对模型进行K折交叉验证，注意scoring可以根据需要改变（其他选择查看API）
# #     a=recall_score(y_test, y_pred,average='micro')
#     recall_score = cross_val_score(svm_model, x_train, y_train, cv=10)
#     # recall_score = cross_val_score(svm_model, x_train, y_train.values.ravel(), scoring='recall', cv=10,average='micro')
#     return recall_score.mean()  # 返回K次交叉验证评分的均值
#
# C_range = np.linspace(100,300,100) # 此处填写参数C的取值范围
# for i in C_range:
#     avg_score = select_c_function(i)
#     print('当C值为{}时，K折交叉验证的平均分是{}'.format(i, avg_score))

#
if __name__=="__main__":
    # load data
    x_train, x_test, y_train, y_test = ms.train_test_split(df[feature_names], df['W'],
                                                           random_state=1, train_size=0.7)
    # training
    cls = svm.SVC(kernel='rbf', C=289.89898989898995,gamma=0.710204081632653)
    cls.fit(x_train.astype('int'), y_train.astype('int'))
    # accuracy
    # print('Test score: %.4f' % cls.score(x_test.astype('int'), y_test.astype('int')))
    # print('Train score: %.4f' % cls.score(x_train.astype('int'), y_train.astype('int')))

    print('Accuracy: %.2f' % cls.score(x_train.astype('int'), y_train.astype('int')))
    # print bad case id
    # bad_idx = find_badcase(x_test.astype('int'),y_test.astype('int'))

    y_pred = cls.predict(x_test)

    x = np.arange(0, 200, 1)
    y1 = y_pred[0:200]
    y2 = y_train[0:200]
    plt.figure()
    matplotlib.rcParams['font.sans-serif'] = ['times new roman']  # 指定中文字体
    plt.plot(x, y1, color='blue', linewidth='1.0', label='Preditict_Data')
    plt.plot(x, y2, color='red', linewidth='1.0', linestyle='--', label='Training_Data')
    plt.xlabel('Number')
    plt.ylabel('Energy Consumption (wh/min)/(1000m3)')
    plt.legend()
    plt.title('HVAC Energy Consumption Predition')
    plt.show()
    print(y_pred.tolist())

    # 绘制决策边界
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制训练数据点
    ax.scatter(x_train['Indoor_Temperature'], x_train['Globe_temperature'], y_train, c='r', label='Training Data')

    # 绘制支持向量
    sv_idx = cls.support_
    ax.scatter(x_train['Indoor_Temperature'].iloc[sv_idx], x_train['Globe_temperature'].iloc[sv_idx],
               y_train.iloc[sv_idx], s=50, c='red', marker='o', edgecolors='g', label='Support Vectors')

    # 绘制决策边界
    x1_min, x1_max = x_train['Indoor_Temperature'].min() - 1, x_train['Indoor_Temperature'].max() + 1
    x2_min, x2_max = x_train['Globe_temperature'].min() - 1, x_train['Globe_temperature'].max() + 1
    x3_min, x3_max = x_train['Humidity'].min() - 1, x_train['Humidity'].max() + 1
    x4_min, x4_max = x_train['Mean_radiant_temperature'].min() - 1, x_train['Mean_radiant_temperature'].max() + 1
    x5_min, x5_max = x_train['Wind_Speed'].min() - 1, x_train['Wind_Speed'].max() + 1
    x6_min, x6_max = x_train['PMV'].min() - 1, x_train['PMV'].max() + 1
    x7_min, x7_max = x_train['Set_Point'].min() - 1, x_train['Set_Point'].max() + 1
    x8_min, x8_max = x_train['Outdoor_T'].min() - 1, x_train['Outdoor_T'].max() + 1

    xx1, xx2, xx3, xx4, xx5, xx6, xx7, xx8 = np.meshgrid(np.arange(x1_min, x1_max, 1),
                                                         np.arange(x2_min, x2_max, 1),
                                                         np.arange(x3_min, x3_max, 1),
                                                         np.arange(x4_min, x4_max, 1),
                                                         np.arange(x5_min, x5_max, 1),
                                                         np.arange(x6_min, x6_max, 1),
                                                         np.arange(x7_min, x7_max, 1),
                                                         np.arange(x8_min, x8_max, 1))

    zz = cls.predict(np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel(), xx4.ravel(),
                           xx5.ravel(), xx6.ravel(), xx7.ravel(), xx8.ravel()])

    zz = zz.reshape(xx1.shape)

    ax.plot_surface(xx1, xx2, zz, alpha=0.2, cmap='viridis')

    ax.set_xlabel('Indoor Temperature')
    ax.set_ylabel('Globe Temperature')
    ax.set_zlabel('Energy Consumption')
    ax.legend()
    plt.title('Decision Boundary')
    plt.show()


