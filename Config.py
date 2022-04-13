class Config(object):
    def __init__(self):
        # 数据集种类
        self.num_classes = 102

        # 是否打印训练进度
        self.print_train_process = True

        # 是否打印测试进度
        self.print_test_process = True

        # 是否保存最好的模型
        self.save_best_model = True

        # 是否接着已有的模型继续训练
        self.continue_train = True

        # 训练epoch
        self.epoch = 102

        # 训练batch_size
        self.batch_size = 8

        # 训练interval
        self.log_interval = 10

        # 线程数
        self.num_workers = 8

        # 默认resnet路径
        self.resnet_path = "./resnet50-19c8e357.pth"

        # 默认model路径
        self.model_path = "pre_models/flowers/models.pth"

        # 默认model_dict路径
        self.model_dict_path = "pre_models/flowers/model_dict.pth"

        # 训练集路径
        self.train_path = ".\home\data\Flowers\\train"

        # 测试集路径
        self.test_path = ".\home\data\Flowers\\test"

        self.alpha = 100

        # 是否绘制图片
        self.draw_pic = False

        self.savepath = ".\ImageSave"
