"""
随机森林数据预处理
处理训练集 验证集 测试集 增加words字段 分词后用空格连接

"""
import pandas as pd
import jieba
from config import Config
# 导入数据
config_obj = Config()
def save_data(input_dir, output_dir):
    train_data = pd.read_csv(input_dir, sep='\t', names=['title', 'label'])
    # 逐词进行分词
    # word_list = []
    # for title in train_data['title'].tolist():
    #     word_list.append(' '.join(jieba.lcut(title)))
    # train_data['words'] = word_list
    # print(train_data)

    # 使用map实现分词
    train_data['words'] = list(map(lambda x: ' '.join(jieba.lcut(x)), train_data['title']))
    # print(train_data)
    # 写入到输出文件
    train_data.to_csv(output_dir, index=False)

if __name__=='__main__':
    # 数据与代码隔离
    path_list = [
        [config_obj.train_path, config_obj.processing_train_path],
        [config_obj.dev_path, config_obj.processing_dev_path],
        [config_obj.test_path, config_obj.processing_test_path]
    ]
    for input_dir, output_dir in path_list:
        save_data(input_dir, output_dir)