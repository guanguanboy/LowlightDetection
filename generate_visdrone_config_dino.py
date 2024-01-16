import mmcv
from mmengine.config import Config
#from mmdet.apis import set_random_seed

# 获取基本配置文件参数
cfg = Config.fromfile('./configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py')

# 修改数据集类型以及文件路径
cfg.dataset_type = 'CocoDataset'
cfg.data_root = '/data/liguanlin/Datasets/Visdrone/'
cfg.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
#cfg.classes = ('ignored regions','pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
#    'awning-tricycle', 'bus', 'motor','others')

cfg.test_dataloader.dataset.type = 'CocoDataset'
cfg.test_dataloader.dataset.data_root = '/data/liguanlin/Datasets/Visdrone/VisDrone2019-DET-test-dev/'
cfg.test_dataloader.dataset.ann_file = '/data/liguanlin/Datasets/Visdrone/annotation_test_10.json'
cfg.test_dataloader.dataset.data_prefix = 'images'
#cfg.data.test.classes = ('ignored regions','pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor','others')
#cfg.test_dataloader.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

cfg.train_dataloader.dataset.type = 'CocoDataset'
cfg.train_dataloader.dataset.data_root = '/data/liguanlin/Datasets/Visdrone/VisDrone2019-DET-train/'
cfg.train_dataloader.dataset.ann_file = '/data/liguanlin/Datasets/Visdrone/annotation_train_10.json'
cfg.train_dataloader.dataset.data_prefix = 'images'
#cfg.data.train.classes = ('ignored regions','pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor','others')
#cfg.train_dataloader.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

cfg.val_dataloader.dataset.type = 'CocoDataset'
cfg.val_dataloader.dataset.data_root = '/data/liguanlin/Datasets/Visdrone/VisDrone2019-DET-val/'
cfg.val_dataloader.dataset.ann_file = '/data/liguanlin/Datasets/Visdrone/annotation_val_10.json'
cfg.val_dataloader.dataset.data_prefix = 'images'
#cfg.data.val.classes = ('ignored regions','pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor','others')
#cfg.val_dataloader.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
# 修改bbox_head中的类别数
cfg.model.bbox_head.num_classes = 10
# 使用预训练好的faster rcnn模型用于finetuning
cfg.load_from = 'checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
# 设置工作目录用于存放log和临时文件
cfg.work_dir = './work_dir_custom'

# 原本的学习率是在8卡基础上训练设置的，现在单卡需要除以8
cfg.optim_wrapper.optimizer.lr = 0.02 / 8
#cfg.optim_wrapper.lr_config.warmup = None
#cfg.log_processor.interval = 10

# 由于是自定义数据集，需要修改评价方法
cfg.val_evaluator.metric = 'bbox'
# 设置evaluation间隔减少运行时间
cfg.val_evaluator.interval = 10
# 设置存档点间隔减少存储空间的消耗
#cfg.checkpoint.interval = 10

# 固定随机种子使得结果可复现
cfg.seed = 0
#set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

train_cfg = dict(max_epochs=10, val_interval=1)

# 打印所有的配置参数
print(f'Config:\n{cfg.pretty_text}')

#mmcv.mkdir_or_exist(F'{cfg.work_dir}')
cfg.dump(F'{cfg.work_dir}/customformat_visdrone_cls_10_dino.py')