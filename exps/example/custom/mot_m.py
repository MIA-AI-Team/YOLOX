from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.67
        self.width = 0.75
        self.data_dir = "datasets/MOT20"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (896, 1600)
        self.test_size = (896, 1600)
        #self.test_size = (736, 1920)
        self.random_size = (20, 36)
        self.max_epoch = 100
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 8.0
        self.warmup_epochs = 1
        self.save_history_ckpt = False