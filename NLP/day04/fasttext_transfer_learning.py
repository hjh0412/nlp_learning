"""
目标：使用fasttext官网提供的模型 做本地推理-迁移学习


"""
import fasttext
import transformers
# 导入模型
model = fasttext.load_model('./models/cc.zh.300.bin')
# 查询中文词向量
vector = model.get_word_vector("音乐")
# 中文词汇临近词查询
vector_near = model.get_nearest_neighbors("音乐")