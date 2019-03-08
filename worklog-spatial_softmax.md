## todo

 - [ ] 复现 simple baseline
 - [x] 修复 ylaunch in nj-site
 - [ ] 添加 ImageNet pretrain
 - [ ] 增加温度 in softmax, 因为增加温度, softmax 更敏感, in - out 差距较小时, loss 就足够小, 好优化
 - [ ] 把 out_cyc pooling 的结果都不传梯度
 - [ ] ~~调大学习率~~
 - [ ] 

## 03.08

p{0, 1}
m{.7, .8, .9}
r{"3,1", "8,4,2,1"}
t{1,50,}



p0 m.7

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 

p.5 m.8 r3,1

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW .5 --probMargin .8 --rs 3,1

p0 m.9

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .9

---
p0 m.7 r3,1

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 --rs 3,1

p0 m.8 r3,1

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .8 --rs 3,1

p0 m.9 r3,1

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .9 --rs 3,1


---
p.5 m.7

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW .5 --probMargin .7 

p.5 m.8

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW .5 --probMargin .8 

p.5 m.9

rlaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW .5 --probMargin .9 

---
---

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0.8




args: p[1, 4, 10]* m[0, .5, .8]

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_msssm_mean.yaml --gpus 0,1,2,3 --pointMaxW 1 --probMargin 0



| pointW \ margin | 0 | 0.8 | .5 |
| - | - |- |- |
| 0 | 81.51 | 84.58 | |
| 1 | 79.20 | 81.76 | 33.00 |
| 4 | 71.49 | 81.06 | 54.82 |
| 10 | 65.54 | 72.88 | 67.85 |




## 03.07  

### avg, pointMax = None:

| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 60.982 | 52.055 | 49.992 | 44.782 | 51.272 | 55.388 | 60.533 | 53.698 | 15.363 |


Epoch: [88][0/174]  

| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 90.518 | 81.403 | 72.132 | 64.639 | 75.056 | 64.556 | 63.344 | 74.163 | 19.872 |

### avg+max, pointMax = None:

| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 69.270 | 71.009 | 69.627 | 67.104 | 54.336 | 79.447 | 72.957 | 67.788 | 17.770 |

| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 93.315 | 90.455 | 84.012 | 76.203 | 82.673 | 76.323 | 68.776 | 82.240 | 20.781 |


### avg+max margin=.8, pointMax = None:

| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 94.884 | 89.181 | 85.938 | 79.939 | 76.839 | 79.810 | 74.941 | 81.509 | 22.061 |


Epoch: [109][0/174]
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 95.123 | 92.408 | 85.666 | 79.631 | 83.417 | 79.850 | 74.871 | 84.580 | 22.860 |
=> saving checkpoint to output/mpii/pose_resnet_50/256x256_d256x3_adam_lr1e-3_msssm_mean
"loss" spend time: 1.177531
