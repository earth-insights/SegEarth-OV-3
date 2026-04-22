_base_ = './base_config.py'

model = dict(
    classname_path='./configs/cls_gid.txt', 
    confidence_threshold=0.1,
    prob_thd=0.1
)

dataset_type = 'GID5Dataset'
data_root = 'data/GID'

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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images',
            seg_map_path='labels'
        ),  
        pipeline=test_pipeline))