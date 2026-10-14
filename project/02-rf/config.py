"""
随机森林的配置文件
训练 测试 验证数据集,标签数据路径
训练 测试 验证数据集 处理保存路径
模型保存路径,预测结果保存路径,停用词数据
"""
class Config():
    def __init__(self):
        self.train_path = '../01-data/train.txt'
        self.test_path = '../01-data/test.txt'
        self.dev_path = '../01-data/dev.txt'
        self.label_path = '../01-data/class.txt'

        # 处理数据保存路径
        self.processing_train_path = 'processing/train.csv'
        self.processing_test_path = 'processing/test.csv'
        self.processing_dev_path = 'processing/dev.csv'

        # 模型保存路径
        self.model_path = './models/'
        self.prediction_result_path = './results.'
        self.stop_words_path = '../01-data/stopwords.txt'

if __name__=='__main__':
    import pandas as pd
    config_obj = Config()
    pd.read_csv(config_obj.train_path,sep='\t')
    pd.read_csv(config_obj.stop_words_path)
    import os
    # 写入模型到指定的路径
    # os.makedirs(config_obj.model_path,exist_ok=True)
    # os.makedirs(config_obj.processing_train_path,exist_ok=True)
    pass