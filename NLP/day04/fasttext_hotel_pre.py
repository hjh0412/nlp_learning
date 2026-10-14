"""
利用fasttext训练酒店文本情感分析

1. 数据格式转换
    输入：sentence,label
    输出：
    __lable__positive,negative
2. 模型训练
3. 模型评估
"""
import pandas as pd
import fasttext
def data_pre(input_file_path, output_file_path):
    """
    将输入的文件转换为用于fasttext预测的文本文件
    """
    data = pd.read_csv(input_file_path, sep='\t')
    # 读取数据，准备好格式
    data = data.dropna(subset = ["sentence","label"])
    sentences = data["sentence"].tolist()
    labels = data["label"].tolist()
    output_list = []
    for i, sentence in enumerate(sentences):
        label = labels[i]
        output_line = f"__label__{'positive' if label else 'negative'} {sentence}\n"
        output_list.append(output_line)
    with open(output_file_path,"w",encoding='utf-8') as f:
        f.writelines(output_list)

# 模型训练
def train():
    model = fasttext.train_supervised("./data/train.txt",autotuneValidationFile="./data/dev.txt",autotuneDuration=90)
    model.save_model("./models/fasttext_model.bin")
def validate(test_dataset):
    model = fasttext.load_model("./models/fasttext_model.bin")
    print(model.test(test_dataset))
if __name__=='__main__':
    # data_pre("./data/dev.tsv", "./data/dev.txt")
    # data_pre("./data/train.tsv", "./data/train.txt")
    train()
    validate("./data/dev.txt")