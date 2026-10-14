from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
# 使用cart来预测泰坦尼克号的生存情况 分类问题
def dm01_carttree()->None:
    # 导入数据
    taitan_pd = pd.read_csv("D:/itheima/ML/机器学习/Day03/05-资料/train.csv",encoding='gbk')
    # 打印标签和特征信息
    print(taitan_pd.head(5))
    print(taitan_pd.info())
    # 选取样本集
    x = taitan_pd[['Pclass','Sex','Age']]
    y = taitan_pd[['Survived']]
    print(x.head(5))
    print(x.info())
    # 缺失值处理
    x['Age'] = x['Age'].fillna(x['Age'].mean())
    # 性别是字符串要热编码
    x = pd.get_dummies(x)
    print(x.head(5))
    print(x.info())
    # 划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    # 特征工程（不需要）
    # 训练模型
    cast_tree = DecisionTreeClassifier()
    # 训练模型
    cast_tree.fit(x_train,y_train)
    # 模型预测
    y_pred = cast_tree.predict(x_test)
    print(f"预测结果:{y_pred}")
    # 准确值测试
    accuracy_score(y_test,y_pred)
    print(f"准确率:{accuracy_score(y_test,y_pred)}")
    # 更详细的分类报告
    myreport = classification_report(y_test,y_pred,target_names=['died','survived'])
    print(myreport)
if __name__ == '__main__':
    dm01_carttree()
    print("胡晋辉")