_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_gf-road.txt',
    prob_thd=0.3,
    confidence_threshold=0.1,
)

# dataset settings
dataset_type = 'GFRoadDataset'
data_root = 'data/GF_LowGradeRoadDataset'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/image',
            seg_map_path='test/label'),
        pipeline=test_pipeline))