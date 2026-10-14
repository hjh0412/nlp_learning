from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
# 记住accuracy_score只能用于离散值的分类问题，不能用于连续值的回归问题
# 使用cast来预测泰坦尼克号的生存情况 分类问题
def dm01_casttree()->None:
    # 导入数据
    data,traget = fetch_openml(name='boston', version=1, return_X_y=True, data_home='../data')
    # 划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(data,traget,test_size=0.2,random_state=42)
    # 特征工程
    # 再一点 决策树是不需要标准化的 因为是分割的特征值点!
    transform = StandardScaler()
    x_train = transform.fit_transform(x_train)
    x_test = transform.transform(x_test)
    # 训练模型
    cast_tree = DecisionTreeRegressor()
    # 训练模型
    cast_tree.fit(x_train,y_train)
    # 模型预测
    y_pred = cast_tree.predict(x_test)
    print(f"预测结果:{y_pred}")
    # 平均值误差
    mean_absolute_error(y_test,y_pred)
    print(f"平均绝对误差:{mean_absolute_error(y_test,y_pred)}")
    # 均方误差
    mean_squared_error(y_test,y_pred)
    print(f"均方误差:{mean_squared_error(y_test,y_pred)}")
    # 均方根误差
    root_mean_squared_error(y_test,y_pred)
    print(f"均方根误差:{root_mean_squared_error(y_test,y_pred)}")
if __name__ == '__main__':
    dm01_casttree()