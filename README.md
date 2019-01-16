# pose-mxnet

## File
- [x] 1. pose_ae
    - [x] 1. symbols
        - [x] 1. posenet_v1 
    - [x] 2. function
        - [x] 1. callback
        - [x] 2. loader
        - [x] 3. train
        - [x] 4. test
        - [x] 5. tester
    - [x] 3. ./* (test, train_end2end)
- [x] 2. common
    - [x] 1. backbone
        - [x] 1. hourglass_v1
    - [x] 2. lib
        - [x] 1. dataset
            - [x] 1. coco_pose
            - [x] 2. imdb (no change)
        - [x] 2. utils
            - [x] 1. load_data
            - [x] 2. load_model (no change)
            - [x] 3. create_logger (no change)
            - [x] 4. lr_scheduler (no change)
        - [x] 3. pose
            - [x] 1. pose (get_pose_batch)
    - [ ] 3. ./*
        - [ ] 1. cpu_metric
        - [x] 2. gpu_metric
        - [x] 3. config
- [x] 3. cfgs/pose_ae
    - [x] 1. yaml



## Feature
- [ ] 1. Train
    - [x] 1. symbol - four stacked hourglasses
    - [x] 2. losses - detection loss & grouping loss
    - [ ] 3. dataset
        - [x] 3.1 MSCOCO
        - [ ] 3.2 MPII
    - [ ] 4. dataloader
        - [x] 4.1 MSCOCO
        - [ ] 4.2 MPII
    - [x] 5. optimizer
    - [x] 6. metric

- [ ] 2. Test
    - [x] 1. symbol - four stacked hourglasses
    - [x] 2. multi-scale test
    - [ ] 3. dataset
        - [x] 3.1 MSCOCO
        - [ ] 3.2 MPII
    - [x] 4. dataloader
        - [x] 4.1 MSCOCO
        - [ ] 4.2 MPII
    - [x] 5. metric
