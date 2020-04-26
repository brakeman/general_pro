# Registry
## 主函数: 从config dict 转化为mmcv.Config module
	def main():
	    args = parse_args()
		# example:args.config = '/retinanet_r101_fpn_1x.py 
		# Config 是mmcv中的Config class, 此处已经完成从 配置文件 retinanet_r101_fpn_1x.py 到 class的封装
	    cfg = Config.fromfile(args.config)
		# 。。。省略其他注册 
		# 此处ipdb显示会自动跳转到mmcv/utils/config.py, 因为其定义了 __getattr__ 方法，cfg.model, cfg.train_cfg, cfg.test_cfg,每当类属性不存在时，会被触发.
	    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

#### DETECTORS 
	from .registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
			       ROI_EXTRACTORS, SHARED_HEADS)



#### cfg.keys 展示
	cfg.keys()
	dict_keys(['pickle', 'pkl_load', 'num_classes', 'model', 'train_cfg', 'test_cfg', 'dataset_type', 'data_root', 'img_norm_cfg', 'train_pipeline', 'test_pipeline', 'data', 'evaluation', 'optimizer', 'optimizer_config', 'lr_config', 'checkpoint_config', 'log_config', 'total_epochs', 'dist_params', 'log_level', 'work_dir', 'load_from', 'resume_from', 'workflow', 'gpus', 'seed'])


### build_detector
	def build_detector(cfg, train_cfg=None, test_cfg=None):
		# cfg 上面说过了，是mmcv.Config module，其主要成分来自于 config dict, 也包括其它后来main() 补充的环境变量之类的；
		# DETECTORS 此处非常关键，它并非只是一个只含有名字的实例化对象，见下面代码段，这个函数要把前者导入后者
		# DETECTORS = Registry('detector')
	    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
	    
### DETECTORS: 这里是/mmdet/models/detectors/cascade_rcnn.py 即每一个模型类 头上都带了个帽子，一旦主函数引用到这个类，则自动注册到DETECTORS中
	# 这个就是..registry.py
	from mmdet.utils import Registry
	BACKBONES = Registry('backbone')
	NECKS = Registry('neck')
	ROI_EXTRACTORS = Registry('roi_extractor')
	SHARED_HEADS = Registry('shared_head')
	HEADS = Registry('head')
	LOSSES = Registry('loss')
	DETECTORS = Registry('detector')

    #这里对应上面
	from ..registry import DETECTORS
	@DETECTORS.register_module
	class CascadeRCNN(BaseDetector, RPNTestMixin):

	    def __init__(self,
	 
		 
### build
	def build(cfg, registry, default_args=None):
	    if isinstance(cfg, list):
		modules = [
		    build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
		]
		return nn.Sequential(*modules)
	    else:
		return build_from_cfg(cfg, registry, default_args)

### build_from_cfg: 此处cfg 是上面说过的实例化后的 config class, registry是实例化后的models类，真的是包含所有的model module 
	def build_from_cfg(cfg, registry, default_args=None):
	    """Build a module from config dict.

	    Args:
		cfg (dict): Config dict. It should at least contain the key "type".
		registry (:obj:`Registry`): The registry to search the type from.
		default_args (dict, optional): Default initialization arguments.

	    Returns:
		obj: The constructed object.
	    """
	    assert isinstance(cfg, dict) and 'type' in cfg
	    assert isinstance(default_args, dict) or default_args is None
	    args = cfg.copy()
	    # obj_type='RetinaNet'
	    obj_type = args.pop('type')
	    if mmcv.is_str(obj_type):
	    # 这里obj_cls就是 最终的pytorch model
		obj_cls = registry.get(obj_type)
		if obj_cls is None:
		    raise KeyError('{} is not in the {} registry'.format(
			obj_type, registry.name))
	    elif inspect.isclass(obj_type):
		obj_cls = obj_type
	    else:
		raise TypeError('type must be a str or valid type, but got {}'.format(
		    type(obj_type)))
	    if default_args is not None:
		for name, value in default_args.items():
		    args.setdefault(name, value)
		# 这里实际上就是在执行model初始化！！见下面
	    return obj_cls(**args)
	    
### RetinaNet： 上述 obj_cls()
	@DETECTORS.register_module
	class RetinaNet(SingleStageDetector):
	    def __init__(self,
			 backbone,
			 neck,
			 bbox_head,
			 train_cfg=None,
			 test_cfg=None,
			 pretrained=None):
		super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
		
					
## 1. 作用：一切模块都要先注册
	from mmdet.utils import Registry
	BACKBONES = Registry('backbone')
	NECKS = Registry('neck')
	ROI_EXTRACTORS = Registry('roi_extractor')
	SHARED_HEADS = Registry('shared_head')
	HEADS = Registry('head')
	LOSSES = Registry('loss')
	DETECTORS = Registry('detector')
	
## 2. source code of registry class

	class Registry(object):

	    def __init__(self, name):
		self._name = name
		self._module_dict = dict()

		# 怎么说，要明白 print(Registry) == print(Registry.__repr__())
	    def __repr__(self):
		format_str = self.__class__.__name__ + '(name={}, items={})'.format(
		    self._name, list(self._module_dict.keys()))
		return format_str

	    @property
	    def name(self):
		return self._name

	    @property
	    def module_dict(self):
		return self._module_dict

	    def get(self, key):
		return self._module_dict.get(key, None)

		# 这一行最主要！ model/*.py文件里面的class定义上都会出现@DETECTORS.register_module;
		# 即其结果会作为参数传入这个函数里面
	    def _register_module(self, module_class, force=False):
		"""Register a module.

		Args:
		    module (:obj:`nn.Module`): Module to be registered.
		"""
		if not inspect.isclass(module_class):
		    raise TypeError('module must be a class, but got {}'.format(
			type(module_class)))
		module_name = module_class.__name__
		if not force and module_name in self._module_dict:
		    raise KeyError('{} is already registered in {}'.format(
			module_name, self.name))
		self._module_dict[module_name] = module_class

	    def register_module(self, cls=None, force=False):
		if cls is None:
		    return partial(self.register_module, force=force)
		self._register_module(cls, force=force)
		return cls


note1: function b 的结果会传入a 

	def a(x):
		if x==2:
			return 4
	@a
	def b():
		return 2
---

### 2. 通过configs/retinanet_r101_fpn_1x.py 理解build

	model = dict(
	    type='RetinaNet',
	    pretrained='torchvision://resnet101',
	    backbone=dict(
		type='ResNet',
		depth=101,
		num_stages=4,
		out_indices=(0, 1, 2, 3),
		frozen_stages=1,
		norm_cfg=dict(type='BN', requires_grad=True),
		style='pytorch'),
	    neck=dict(
		type='FPN',
		in_channels=[256, 512, 1024, 2048],
		out_channels=256,
		start_level=1,
		add_extra_convs=True,
		num_outs=5),
	    bbox_head=dict(
		type='RetinaHead',
		num_classes=81,
		in_channels=256,
		stacked_convs=4,
		feat_channels=256,
		octave_base_scale=4,
		scales_per_octave=3,
		anchor_ratios=[0.5, 1.0, 2.0],
		anchor_strides=[8, 16, 32, 64, 128],
		target_means=[.0, .0, .0, .0],
		target_stds=[1.0, 1.0, 1.0, 1.0],
		loss_cls=dict(
		    type='FocalLoss',
		    use_sigmoid=True,
		    gamma=2.0,
		    alpha=0.25,
		    loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
	# training and testing settings
	train_cfg = dict(
	    assigner=dict(
		type='MaxIoUAssigner',
		pos_iou_thr=0.5,
		neg_iou_thr=0.4,
		min_pos_iou=0,
		ignore_iof_thr=-1),
	    allowed_border=-1,
	    pos_weight=-1,
	    debug=False)
	test_cfg = dict(
	    nms_pre=1000,
	    min_bbox_size=0,
	    score_thr=0.05,
	    nms=dict(type='nms', iou_thr=0.5),
	    max_per_img=100)
	# dataset settings
	dataset_type = 'CocoDataset'
	data_root = 'data/coco/'
	img_norm_cfg = dict(
	    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
	train_pipeline = [
	    dict(type='LoadImageFromFile'),
	    dict(type='LoadAnnotations', with_bbox=True),
	    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
	    dict(type='RandomFlip', flip_ratio=0.5),
	    dict(type='Normalize', **img_norm_cfg),
	    dict(type='Pad', size_divisor=32),
	    dict(type='DefaultFormatBundle'),
	    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
	]
	test_pipeline = [
	    dict(type='LoadImageFromFile'),
	    dict(
		type='MultiScaleFlipAug',
		img_scale=(1333, 800),
		flip=False,
		transforms=[
		    dict(type='Resize', keep_ratio=True),
		    dict(type='RandomFlip'),
		    dict(type='Normalize', **img_norm_cfg),
		    dict(type='Pad', size_divisor=32),
		    dict(type='ImageToTensor', keys=['img']),
		    dict(type='Collect', keys=['img']),
		])
	]
	data = dict(
	    imgs_per_gpu=2,
	    workers_per_gpu=2,
	    train=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/instances_train2017.json',
		img_prefix=data_root + 'train2017/',
		pipeline=train_pipeline),
	    val=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/instances_val2017.json',
		img_prefix=data_root + 'val2017/',
		pipeline=test_pipeline),
	    test=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/instances_val2017.json',
		img_prefix=data_root + 'val2017/',
		pipeline=test_pipeline))
	evaluation = dict(interval=1, metric='bbox')
	# optimizer
	optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
	optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
	# learning policy
	lr_config = dict(
	    policy='step',
	    warmup='linear',
	    warmup_iters=500,
	    warmup_ratio=1.0 / 3,
	    step=[8, 11])
	checkpoint_config = dict(interval=1)
	# yapf:disable
	log_config = dict(
	    interval=50,
	    hooks=[
		dict(type='TextLoggerHook'),
		# dict(type='TensorboardLoggerHook')
	    ])
	# yapf:enable
	# runtime settings
	total_epochs = 12
	dist_params = dict(backend='nccl')
	log_level = 'INFO'
	work_dir = './work_dirs/retinanet_r101_fpn_1x'
	load_from = None
	resume_from = None
	workflow = [('train', 1)]
		
