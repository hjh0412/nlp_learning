"""
熟悉ngram特征机制
构建词的索引列表
根据ngram机制构建新的列表
"""
input_list = [1,3,2,1,5,3]
ngram_range = 5
# 方法一
n_gram_words = []
# 长度是n - ngram_range + 1
for i in range(len(input_list) - ngram_range + 1):
    n_gram_words.append(input_list[i : i + ngram_range])
print(n_gram_words)                                  #[[1, 3, 2, 1, 5], [3, 2, 1, 5, 3]]
# 使用方法二 zip进行处理
print(list(zip([1,2,3,4],[10,23,56,82],[2,5,6])))    #[(1, 10, 2), (2, 23, 5), (3, 56, 6)]
print([input_list[i : ] for i in range(ngram_range)]) #[[1, 3, 2, 1, 5, 3], [3, 2, 1, 5, 3], [2, 1, 5, 3], [1, 5, 3], [5, 3]]
# 使用zip加解包的方式实现代码
print(list(zip(*[input_list[i : ] for i in range(ngram_range)]))) #[(1, 3, 2, 1, 5), (3, 2, 1, 5, 3)]