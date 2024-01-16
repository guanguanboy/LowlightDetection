_base_ = '../configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'
model = dict(bbox_head=dict(num_classes=10))
 
dataset_type = 'CocoDataset'
data_root = '/data/liguanlin/Datasets/Visdrone/'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
 
# Modify dataset related settings
metainfo = {
    'classes': ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'),
 
}
#backend_args = None
 
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotation_train_10.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/'),
        ))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotation_val_10.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
        ))
 
val_evaluator = dict(ann_file=data_root + 'annotation_val_10.json')
 
# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotation_test_10.json',
        data_prefix=dict(img='VisDrone2019-DET-test-dev/images/'),
        test_mode=True,
        ))
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'annotation_test_10.json',
    outfile_prefix='./work_dir_custom/visdrone_detection/test')
 
#test_dataloader = val_dataloader
#test_evaluator = val_evaluator
        
evaluation = dict(interval=5, metric='bbox', classwise=True)  
load_from = './checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'