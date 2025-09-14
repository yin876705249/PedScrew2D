_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=350, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=500,
        milestones=[250, 300, 330],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
# default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
default_hooks = dict(checkpoint=dict(save_best='OKS', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='PoseWithSegmentation',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    # neck=dict(
    #     type='FPN',  # 使用特征金字塔网络作为例子
    #     in_channels=[32, 64, 128, 256],  # 假设从HRNet的各个阶段接收的通道数
    #     out_channels=256,  # 统一输出通道数
    #     num_outs=4  # 输出的特征层次数
    # ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=10,  # number of keypoints
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    seg_head=dict(
        type='FCNHead',
        in_channels=32,
        in_index=0,
        channels=256,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=4,
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1),
            dict(type='DiceLoss', loss_weight=0.1)
        ]
    ),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        output_heatmaps=True,
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'SpineDataset'
data_mode = 'topdown'
data_root = '/workspace/Spine/data'

# pipelines
train_pipeline = [
    dict(type='LoadImageAndMask', to_float32=True, mask_to_float32=False, mask_color_type='unchanged', apply_distance_transform=False),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='SpineTopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputsWithMask')
]
val_pipeline = [
    dict(type='LoadImageAndMask', to_float32=True, mask_to_float32=False, mask_color_type='unchanged', apply_distance_transform=False),
    dict(type='GetBBoxCenterScale'),
    dict(type='SpineTopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputsWithMask')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # ann_file='annotations/spine_2d_train.json',
        data_prefix=dict(img='images/', seg='masks/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # ann_file='annotations/spine_2d_val.json',
        # bbox_file=None,
        data_prefix=dict(img='images/', seg='masks/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # ann_file='annotations/spine_2d_test.json',  # Change to your val annotation file
        # bbox_file=None,
        data_prefix=dict(img='images/', seg='masks/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
# evaluators
# val_evaluator = dict(
#     type='SpineMetric',
#     ann_file=data_root + '/annotations/spine_2d_val.json')
val_evaluator = [
    dict(
        type='ScrewMetric',
        skeleton_info=[
            (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
        ],
        sigmas = [0.25, 0.25, 0.25, 0.25, 0.70, 0.70, 0.70, 0.70, 0.75, 0.75]
    ),
    dict(type='PCKAccuracy', thr=0.05),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]),
    dict(type='EPE'),
    ]
test_evaluator = val_evaluator
