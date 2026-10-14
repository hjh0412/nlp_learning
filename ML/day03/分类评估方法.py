"""
分类评估的几种方法：
参数： 准确率，精准率，召回率，F1_score
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
# 混淆矩阵 得到 真正例 伪假例 伪真例 真假例 之间的值

def dm01_confusion_matrix():    
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本  
    y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']   
    labels = ['恶性', '良性']    
    dataframe_labels = ['恶性(正例)', '良性(反例)'] 
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本   
    print("模型A:")   
    print("-" * 13)    
    y_pred1= ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性'] 
    result = confusion_matrix(y_true, y_pred1, labels=labels)
    print(pd.DataFrame(result, columns=dataframe_labels,  index=dataframe_labels))
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    print('模型B:')   
    print('-' * 13)   
    y_pred2= ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性'] 
    # 混淆矩阵函数的参数 参数一：真实值 参数二：预测值 参数三：标签的顺序  
    result = confusion_matrix(y_true, y_pred2, labels=labels)   
    print(pd.DataFrame(result,columns=dataframe_labels, index=dataframe_labels))

# 计算分类评估的精准度
def dm02_precision_score():  
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本    
    y_true = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性", "良性", "良性", "良性"]  
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本  
    y_pred1 = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]   
    result = precision_score(y_true, y_pred1, pos_label="恶性")  
    result1 = accuracy_score(y_true, y_pred1)
    print("模型A精度:", result)   
    print("模型A准确率:", result1)
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本   
    y_pred2 = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性",  "恶性", "恶性", "恶性", "良性"]  
    result = precision_score(y_true, y_pred2, pos_label="恶性")   
    result2 = accuracy_score(y_true, y_pred2)
    print("模型B精度:", result)
    print("模型B准确率:", result2)


# 计算模型评估的召回率
def dm03_recall_score():    
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本   
    y_true = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性",  "良性", "良性", "良性", "良性"] 
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    y_pred1 = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]  
    result = recall_score(y_true, y_pred1, pos_label="恶性")    
    print("模型A召回率:", result)  
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本  
    y_pred2= ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性","恶性", "恶性", "恶性", "良性"]  
    result = recall_score(y_true, y_pred2, pos_label="恶性")   
    print("模型B召回率:", result)

def dm04_F1():    
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性", "良性", "良性", "良性"]  
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    y_pred = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]  
    result = f1_score(y_true, y_pred, pos_label="恶性")  
    print("模型Af1-score:", result)  
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本  
    y_pred = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性",  "恶性", "恶性", "恶性", "良性"]  
    result = f1_score(y_true, y_pred, pos_label="恶性")
    print("模型Bf1-score::", result)


if __name__ == '__main__':
    dm01_confusion_matrix()
    dm02_precision_score()
    dm03_recall_score()
    dm04_F1()