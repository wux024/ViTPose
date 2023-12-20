_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/animalpose.py',
    '../../../../_base_/datasets/ap10k_info.py',
    '../../../../_base_/datasets/apt36k.py',
    '../../../../_base_/datasets/atrw.py',
    '../../../../_base_/datasets/awa.py',
    '../../../../_base_/datasets/fly.py',
    '../../../../_base_/datasets/horse10.py',
    '../../../../_base_/datasets/locust.py',
    '../../../../_base_/datasets/macaque.py',
    '../../../../_base_/datasets/standfordextra.py',
    '../../../../_base_/datasets/zebra.py',
]
evaluation = dict(interval=10, metric=['PCK', 'AUC', 'EPE'], save_best='AUC')

optimizer = dict(type='AdamW',
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0.1,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=12, 
                                    layer_decay_rate=0.8,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                    )
                )

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
target_type = 'GaussianHeatmap'
animalpose_channel_cfg = dict(
    num_output_channels=20,
    dataset_joints=20,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ])
ap10k_channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
apt36k_channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
atrw_channel_cfg = dict(
    num_output_channels=15,
    dataset_joints=15,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    ])
awa_channel_cfg = dict(
    num_output_channels=39,
    dataset_joints=39,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38
    ])
fly_channel_cfg = dict(
    num_output_channels=32,
    dataset_joints=32,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ])
horse10_channel_cfg = dict(
    num_output_channels=22,
    dataset_joints=22,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 21
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        21
    ])
locust_channel_cfg = dict(
    num_output_channels=35,
    dataset_joints=35,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
    ])
macaque_channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
standfordextra_channel_cfg = dict(
    num_output_channels=20,
    dataset_joints=20,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ])
zebra_channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ])
# model settings
model = dict(
    type='TopDownMoE',
    pretrained=None,
    backbone=dict(
        type='ViTMoE',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        num_expert=6,
        part_features=192
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=animalpose_channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    associate_keypoint_head=[
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=ap10k_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=apt36k_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=atrw_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=awa_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=fly_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=horse10_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=locust_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=macaque_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=standfordextra_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=zebra_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        ],
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

animalpose_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=animalpose_channel_cfg['num_output_channels'],
    num_joints=animalpose_channel_cfg['dataset_joints'],
    dataset_channel=animalpose_channel_cfg['dataset_channel'],
    inference_channel=animalpose_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=0,
)

ap10k_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=ap10k_channel_cfg['num_output_channels'],
    num_joints=ap10k_channel_cfg['dataset_joints'],
    dataset_channel=ap10k_channel_cfg['dataset_channel'],
    inference_channel=ap10k_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=1,
)

apt36k_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=apt36k_channel_cfg['num_output_channels'],
    num_joints=apt36k_channel_cfg['dataset_joints'],
    dataset_channel=apt36k_channel_cfg['dataset_channel'],
    inference_channel=apt36k_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=2,
)
atrw_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=atrw_channel_cfg['num_output_channels'],
    num_joints=atrw_channel_cfg['dataset_joints'],
    dataset_channel=atrw_channel_cfg['dataset_channel'],
    inference_channel=atrw_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=3,
)
awa_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=awa_channel_cfg['num_output_channels'],
    num_joints=awa_channel_cfg['dataset_joints'],
    dataset_channel=awa_channel_cfg['dataset_channel'],
    inference_channel=awa_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=4,
)
fly_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=fly_channel_cfg['num_output_channels'],
    num_joints=fly_channel_cfg['dataset_joints'],
    dataset_channel=fly_channel_cfg['dataset_channel'],
    inference_channel=fly_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=5,
)
horse10_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=horse10_channel_cfg['num_output_channels'],
    num_joints=horse10_channel_cfg['dataset_joints'],
    dataset_channel=horse10_channel_cfg['dataset_channel'],
    inference_channel=horse10_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=6,
)
locust_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=locust_channel_cfg['num_output_channels'],
    num_joints=locust_channel_cfg['dataset_joints'],
    dataset_channel=locust_channel_cfg['dataset_channel'],
    inference_channel=locust_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=7,
)
macaque_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=macaque_channel_cfg['num_output_channels'],
    num_joints=macaque_channel_cfg['dataset_joints'],
    dataset_channel=macaque_channel_cfg['dataset_channel'],
    inference_channel=macaque_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=8,
)
standfordextra_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=standfordextra_channel_cfg['num_output_channels'],
    num_joints=standfordextra_channel_cfg['dataset_joints'],
    dataset_channel=standfordextra_channel_cfg['dataset_channel'],
    inference_channel=standfordextra_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=9,
)
zebra_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=zebra_channel_cfg['num_output_channels'],
    num_joints=zebra_channel_cfg['dataset_joints'],
    dataset_channel=zebra_channel_cfg['dataset_channel'],
    inference_channel=zebra_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=39,
    dataset_idx=10,
)
animalpose_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
ap10k_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
apt36k_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
atrw_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs', 'dataset_idx'
        ]),
]
awa_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
fly_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
horse10_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
locust_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
macaque_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
standfordextra_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
zebra_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'dataset_idx'
        ]),
]
test_pipeline = val_pipeline

animalpose_data_root = 'data/coco'
ap10k_data_root = 'data/ap10k'
apt36k_data_root = 'data/apt36k'
atrw_data_root = 'data/atrw'
awa_data_root = 'data/awa2'
fly_data_root = 'data/fly'
horse10_data_root = 'data/horse10'
locust_data_root = 'data/locust'
macaque_data_root = 'data/macaque'
standfordextra_data_root = 'data/standfordextra'
zebra_data_root = 'data/zebra'


data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=[
        dict(
            type='AnimalPoseDataset',
            ann_file=f'{animalpose_data_root}/annotations/animalpose_train.json',
            img_prefix=f'{animalpose_data_root}/',
            data_cfg=animalpose_data_cfg,
            pipeline=animalpose_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalAP10KDataset',
            ann_file=f'{ap10k_data_root}/annotations/ap10k_annotations_train.json',
            img_prefix=f'{ap10k_data_root}/data/',
            data_cfg=ap10k_data_root,
            pipeline=ap10k_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalAP10KDataset',
            ann_file=f'{apt36k_data_root}/annotations/mpii_train.json',
            img_prefix=f'{apt36k_data_root}/images/',
            data_cfg=apt36k_data_cfg,
            pipeline=apt36k_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalATRWDataset',
            ann_file=f'{atrw_data_root}/annotations/keypoint_train.json',
            img_prefix=f'{atrw_data_root}/data/',
            data_cfg=atrw_data_cfg,
            pipeline=atrw_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalAWADataset',
            ann_file=f'{awa_data_root}/annotations/awa2_annotations_train.json',
            img_prefix=f'{awa_data_root}/data/',
            data_cfg=awa_data_cfg,
            pipeline=awa_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalFlyDataset',
            ann_file=f'{fly_data_root}/annotations/fly_train.json',
            img_prefix=f'{fly_data_root}/images/',
            data_cfg=fly_data_cfg,
            pipeline=fly_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalHorse10Dataset',
            ann_file=f'{horse10_data_root}/annotations/horse10-train-split3.json',
            img_prefix=f'{horse10_data_root}/',
            data_cfg=horse10_data_cfg,
            pipeline=horse10_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalLocustDataset',
            ann_file=f'{locust_data_root}/annotations/locust_train.json',
            img_prefix=f'{locust_data_root}/images/',
            data_cfg=locust_data_cfg,
            pipeline=locust_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalMacaqueDataset',
            ann_file=f'{macaque_data_root}/annotations/macaque_train.json',
            img_prefix=f'{macaque_data_root}/images/',
            data_cfg=macaque_data_cfg,
            pipeline=macaque_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalStandfordExtraDataset',
            ann_file=f'{standfordextra_data_root}/annotations/StanfordExtra_annotations_train.json',
            img_prefix=f'{standfordextra_data_root}/',
            data_cfg=standfordextra_data_cfg,
            pipeline=standfordextra_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='AnimalZebraDataset',
            ann_file=f'{zebra_data_root}/annotations/zebra_train.json',
            img_prefix=f'{zebra_data_root}/images/',
            data_cfg=zebra_data_cfg,
            pipeline=zebra_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        ],
    val=dict(
        type='AnimalAP10KDataset',
        ann_file=f'{apt36k_data_root}/annotations/apt36k_annotations_val.json',
        img_prefix=f'{apt36k_data_root}/data/',
        data_cfg=apt36k_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='AnimalAP10KDataset',
        ann_file=f'{apt36k_data_root}/annotations/apt36k_annotations_test.json',
        img_prefix=f'{apt36k_data_root}/data/',
        data_cfg=apt36k_data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

