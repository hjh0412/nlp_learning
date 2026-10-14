"""
1.声明数据路径
2.fasttext训练调用
3.模型保存
4.模型推理
"""
# 导包
import fasttext
# 声明训练数据的路径
data_dir = "./data/bj30ad"
# 无监督的训练
model = fasttext.train_unsupervised(data_dir)
# 保存模型
model.save_model("./model/fasttext_train.bin")