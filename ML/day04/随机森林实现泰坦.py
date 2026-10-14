import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def dm01_randomforest()->None:
    # 导入数据
    taitan_pd = pd.read_csv("D:/itheima/ML/机器学习/Day03/05-资料/train.csv",encoding='gbk')
    # 打印标签和特征信息
    print(taitan_pd.head(5))
    print(taitan_pd.info())
    # 选取样本集
    # X是训练集是一个二维的数组
    x = taitan_pd[['Pclass','Sex','Age']].copy()
    # y作为标签是一个一维的数组
    y = taitan_pd['Survived']
    # 缺失值处理
    x['Age'] = x['Age'].fillna(x['Age'].mean())
    # 性别是字符串要热编码
    x = pd.get_dummies(x)
    # 特征工程
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=22)
    # 模型训练
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    # 模型预测
    y_pred = model.predict(x_test)
    print(y_pred)
    score = model.score(x_test,y_test)
    print(f"model模型评分为：{score}")
    # 使用交叉验证和网格搜素对模型进行评估
    estimate = RandomForestClassifier()
    param = {"n_estimators":[10,20,30,40,50,60,70,80,90,100],"max_depth":[2,4,6,8,10]}
    grid_search = GridSearchCV(estimator=estimate, param_grid=param, cv=5)
    grid_search.fit(x_train,y_train)
    print(f'交叉验证中最好的结果：{grid_search.best_score_}')
    print(f'最好的参数模型：{grid_search.best_estimator_}')
    print(f"grid_search模型评分为：{grid_search.score(x_train,y_train)}")
if __name__=="__main__":
    dm01_randomforest()