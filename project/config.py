"""
配置类
训练集
测试集
验证集
标签

"""
class ConFig():
    def __init__(self):
        self.train_data_path = './01-data/train.txt'
        self.test_data_path = './01-data/test.txt'
        self.dev_data_path = './01-data/dev.txt'
        self.class_data_path = './01-data/class.txt'

        