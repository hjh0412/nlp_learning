"""
投满分项目配置文件
配置数据集路径，包括
train.txt -- 训练集
test.txt -- 测试集
dev.txt -- 验证集
class.txt -- 标签数据
"""

class Config(object):
    def __init__(self):
        """
        配置4个文件路径
        """
        # 训练集
        self.train_data_path = r"D:\itheima\toumanfen\03-代码\01-data\train.txt"
        # 测试集
        self.test_data_path = './test.txt'
        # 验证集
        self.dev_data_path = './dev.txt'
        # 标签数据
        self.class_data_path = './class.txt'

if __name__ == '__main__':
    config_obj = Config()
    train_path = config_obj.train_data_path
    # 读取训练数据的相关信息
    with open(train_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    print(line_count)
    result= config_obj.train_data_path == r"D:\itheima\toumanfen\03-代码\01-data\train.txt"
    print(result)
    # assert函数后面的值是bool类型的数据 如果为false就会报错 常常用于检验
    assert result
    print('测试通过！')