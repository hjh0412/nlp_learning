"""
学习句子截断与补齐
定义语句数据
执行截断和补齐的操作
"""
from tensorflow.keras.preprocessing import sequence             # 可以帮助进行裁断和补齐
def sentence_padding(sentences:list, cutlen:int):
    # 存储裁断和补充后的语句
    output = []
    for sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len > cutlen:
            # 截断
            output.append(sentence[:cutlen])
        elif sentence_len < cutlen:
            # 补充
            output.append(sentence + [0]*(cutlen - sentence_len))
        else:
            output.append(sentence)
    return output
if __name__=='__main__':
    # 语句的长度
    cutlen = 10
    # 要进行语句处理的语句
    x_train = [
        [1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
        [2, 32, 1, 23, 1],
    ]
    print(len(x_train[0]))
    output = sentence_padding(x_train, cutlen)
    print(output)

    print("*"*100)
    # 法二、使用sequence方法 截断和补齐默认是从前面开始 padding是补齐的意思
    print(sequence.pad_sequences(x_train, maxlen=cutlen, padding='post', truncating='post', value=0))