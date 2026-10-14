"""
学习transformers库的管道模式
情感分析
特征提取
完形填空
阅读理解
ner
"""
from transformers import pipeline
import numpy as np
def sentiment_analysis():
    """
    情感分析
    输入：一个文本
    输出：一个情感标签
    label越高关联越强
    """
    model_dir = r"PretrainedModel\chinese_sentiment"
    clf = pipeline(task="sentiment-analysis", model=model_dir)
    print(clf("人生该如何起头"))

def feature_extraction():
    """
    特征提取
    输入：一个文本
    输出：这个语句的特征向量
    可以通过转换为数组的形式查看这个特征的大小
    """
    model_dir = r"PretrainedModel\bert-base-chinese" 
    ext = pipeline(task="feature-extraction",model=model_dir)
    out = ext("人生该如何起头")
    print(np.array(out).shape)
    # print(out)
def fill_mask():
    """
    完形填空
    输入：带有一个或者多个MASK的文本
    输出：一个包含填词结果的列表
    """
    model_dir = r"PretrainedModel\chinese-bert-wwm" 
    fill = pipeline(task="fill-mask",model=model_dir)
    # 预测单个MASK
    # print(fill("我明天想去[MASK]家吃饭",top_k=1))
    # 预测多个MASK
    sentence = "今天[MASK]的天气[MASK]，所以很适合出去玩"
    new_seq = fill(sentence,top_k=1)[0][0]['sequence']         # 取出第一个词的填词结果
    print(fill(new_seq,top_k=1))
    # seq = "我在黑马程序员[MASK],他们教的[MASK],帮助我拿[MASK]"
    # new_str1 = fill(seq, top_k=1)[0][0]['token_str']
    # seq = seq.replace("[MASK]",new_str1,1)
    # new_str2 = fill(seq,top_k=1)[0][0]['token_str']
    # print(fill(seq.replace("[MASK]",new_str2,1),top_k=1))
    # print(new_str2)
    # print(seq)

def question_answering():
    model_dir = r"PretrainedModel\chinese_pretrain_mrc_roberta_wwm_ext_large"
    qa = pipeline(task="text-generation", model=model_dir)
    context = "我叫张三，我是一个程序员，我的喜好是打篮球"
    question = "我是谁？"
    print(qa(f"上下文：{context};问题{question}；答案："))

def summarization():
    model_dir = r"./qwen3.5-0.8b"
    summarizer = pipeline(task="text-generation", model=model_dir)
    text = (
        "BERT is a transformers model pretrained on a large corpus of English data "
        "in a self-supervised fashion. This means it was pretrained on the raw texts only, "
        "with no humans labelling them in any way. It was pretrained with two objectives: "
        "masked language modeling, and next sentence prediction."
    )
    # result = summarizer(text, max_length=60, min_length=10, truncation=True)
    prompt = f"你是一个文本摘要专家，请你根据下面的英文总结为一个英文摘要，10-60个词：\n{text}"
    print(summarizer(prompt))

def ner():
    model_dir = r"PretrainedModel\roberta-base-finetuned-cluener2020-chinese"
    recognizer = pipeline(task="ner", model=model_dir, aggregation_strategy="simple")
    print(recognizer("我爱北京天安门，天安门上太阳升"))

# 字符串替换函数
def replace(orig_sentence,old,new,count):
    old_len = len(old)
    current_counts = 0
    start_ind = 0
    # "abcd" ab->A
    output_list = []
    for i in range(len(orig_sentence) - old_len + 1):
        if current_counts == count:
            break
        if orig_sentence[i:i+old_len] == old:
            #
            output_list.append(orig_sentence[start_ind:i])
            output_list.append(new)
            current_counts += 1
            # 更新start index
            start_ind = i+old_len
            pass
    output_list.append(orig_sentence[start_ind:])

    return "".join(output_list)

if __name__=='__main__':
    # sentiment_analysis()
    # feature_extraction()
    # fill_mask()
    # question_answering()
    # summarization()
    ner()
    pass