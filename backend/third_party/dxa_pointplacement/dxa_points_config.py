# this file should be placed in mmpose/mmpose/configs/_base_ when mmpose was built from source
# based off of the topdown heatmap config provided by mmpose

import cv2
from mmengine.hooks import EarlyStoppingHook, CheckpointHook
from mmengine.hooks import (DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook, SyncBuffersHook)
from mmengine.runner import LogProcessor
from mmengine.visualization import LocalVisBackend

from mmpose.engine.hooks import PoseVisualizationHook
from mmpose.visualization import PoseLocalVisualizer

# adding parameters decided on through optuna run
optim_wrapper = dict(optimizer=dict(type='Adam',lr=9e-5))

# incidentals
default_scope = 'mmpose'
launcher = 'none'

# runtime
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=20)
val_cfg = dict()
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, by_epoch=True),  # warm-up
    dict(type='ReduceOnPlateauLR', monitor='PCK', verbose=True, begin=100)]

# codec settings
codec = codec = dict(type='MSRAHeatmap',
                     input_size=(654, 1914),
                     heatmap_size=(168, 480), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[67.29219728, 67.29219728, 67.29219728],
        std=[41.70741071, 41.70741071, 41.70741071],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=152,
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=2048,
        out_channels=105,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        shift_heatmap=True,),
    init_cfg=dict(type='Pretrained',
                     checkpoint='https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288-eab8caa8_20201004.pth'))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '' # base path to the folders holding your dataset here

# pipelines
train_pipeline = [
    dict(type='LoadImage', to_float32 =True),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform', shift_prob=0.7, rotate_prob=0.7,
         rotate_factor=16.0, scale_prob=0.7),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

test_pipeline = [
    dict(type='LoadImage', to_float32 =True),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

work_dir = '' # fill in with path you'd like checkpoints saved to

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file='../datasets/datasets/dxa_points.py'), # path to meta file from this repo
        ann_file='', # path to your COCO format json file with DXA annotations
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file='../datasets/datasets/dxa_points.py'), # path to meta file from this repo
        ann_file='', # path to your COCO format json file with DXA annotations
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

# # hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', rule='greater',
                     interval=5, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# logger
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

# file I/O backend
backend_args = dict(backend='local')

custom_hooks = []

# evaluators
val_evaluator = [dict(type='PCKAccuracy', thr=0.1), dict(type='EPE'),
                 dict(type='NME', norm_mode = 'use_norm_item', norm_item='bbox_size')]
test_evaluator = val_evaluator
