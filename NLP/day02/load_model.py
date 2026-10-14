"""
学习如何加载fasttext模型

模型运行是在内存 而存储是在硬盘
"""
import fasttext
model = fasttext.load_model("./model/fasttext_model.bin")
print(model.get_word_vector("music"))