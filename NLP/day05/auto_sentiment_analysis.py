"""
根据transformers自动模型进行文本分类

实例化分词器和模型
分词向量化
把向量传入模型得到结果
"""
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch
def text_classification():
    model_dir = r"PretrainedModel\chinese_sentiment"
    # 实例化分词器
    my_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 示例化模型
    my_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # 定义要进行情感预测的句子
    sentence = "人生何处不青山"
    # 分词向量化
    msg_tensor = my_tokenizer(
        text=sentence,return_tensor='pt',
        padding="max_length",truncation=True,max_length=5,
    )
    print(msg_tensor)
    # 将向量嵌套成张量 先编码
    msg_list = my_tokenizer.encode(
        text=sentence,padding="max_length",
        truncation=True,max_length=5,
    )
    msg_tensor2 = torch.tensor([msg_list])
    my_model.eval()
    # 模型训练输出结果
    result = my_model(msg_tensor2)
    result = my_model(msg_tensor2, return_dict=False)
    result_index = torch.argmax(result[0])
    print(f"最终分类:star {result_index + 1}")
if __name__=='__main__':
    text_classification()
    pass