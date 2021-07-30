import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns   

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



# read file 
data_train = pd.read_csv("./train.csv")
# basic information of saleprice e.g. mean, std, min
data_train['SalePrice'].describe()
# sale price distribution
sns.distplot(data_train['SalePrice'])
# to show !!!!
#plt.show()
'''
Kurtosis=0 与正态分布的陡缓程度相同
Kurtosis>0 比正态分布的高峰更加陡峭
Kurtosis<0 比正态分布的高峰来得平台
Skewness=0 分布形态与正态分布偏度相同
Skewness>0 正偏差数值较大，为正偏或右偏。长尾巴拖在右边。
Skewness<0 负偏差数值较大，为负偏或左偏。长尾巴拖在左边。
'''
data_train['SalePrice'].skew()
data_train['SalePrice'].kurt()

# ----------- basic analyse ----------------
# CentralAir
var = 'CentralAir'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
#print(data)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
#plt.show()

# OverallQual
var = 'OverallQual'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
#print(data)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# YearBuilt  scatter
var = 'YearBuilt'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
#plt.show()


# Neighborhood
var = 'Neighborhood'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# ----------- scientific analyse ----------------
# 生成数值型关系矩阵
#计算每两列之间的相关系数
corrmat = data_train.corr()
#fig, axes = plt.subplots(23)：即表示一次性在figure上创建成2*3的网格
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)


# 生成数值型&离散型关系矩阵
from sklearn import preprocessing
f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    data_train[x] = label.fit_transform(data_train[x])
#print(data_train['Neighborhood'])
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)


# 房价关系矩阵
k  = 10 # 关系矩阵中将显示10个特征
# corrmat.nlargest(k, 'SalePrice')
# 前十关联

# corrmat.nlargest(k, 'SalePrice')['SalePrice']
# ['SalePrice'] 列
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)  # ???
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()


# ----------- 模拟数据 ----------------
from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from  sklearn.model_selection import train_test_split
import numpy as np

# 获取数据
cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = data_train[cols].values
y = data_train['SalePrice'].values
# preprocessing.StandardScaler() 数据标准化，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
# fit_transform()先拟合数据，再标准化
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1)) # 只有一列
X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
