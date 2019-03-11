## todo

 - [x] 复现 simple baseline
 - [x] 修复 ylaunch in nj-site
 - [x] 添加 ImageNet pretrain
 - [x] 增加温度 in softmax, 因为增加温度, softmax 更敏感, in - out 差距较小时, loss 就足够小, 好优化
 - [ ] 把 out_cyc pooling 的结果都不传梯度
 - [ ] ~~调大学习率~~
 - [x] 优化器 adam => sgdm
 - [x] 实验 r4321
 - [x] align center of cyc
 - [ ] 使用分割的 softmax 2d 替换掉 GAP, GMP (更加科学的梯度作用域)
 - [ ] 考虑理论上加 large margin 的可能性
 - [ ] 
 - [ ] 


## 拷问

 * 理论上, 为什么 msssm 会比 mse + gaussian 好?
 * 理论上, 为什么加了 pointMax 会掉点? 为什么单纯的 pointMax 不会收敛到足够的好?
 * 为什么大 temper 表现更好?
 * prob margin 和 temper 对 msssm 的影响是什么? 深层次上等效吗?
 * multi scale 是越细密越好吗? r8421 better than r31,  那么 r4321 如何?
 * 实验中, 为什么我们的 mean@test 极不稳定, 而 Baseline 却很稳定?
 * 为什么 sgdm 在 mse 和 msssm 上都这么弱?

## 03.10

1. 分析实验结果
1. softmax norma, 后加大 temper
1. 添加 prob split
1. 大 temper 下的 loss

### pose @ wh (big temper on baseline p0m0rs8421t13 mse) after sub softmax


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 15

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 20

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 30


---


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 50



ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 75






---

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 12



ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 14



------
------




## 03.09

t 大于 14 则 nan

### sgdpose @ wh (sgd on baseline p0m0rs8421t10 lr-1)

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-1.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 10

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-2.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 10

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-3.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 10


---
ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-1.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 10 --rs 4,3,2,1

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-1.yaml --gpus 0,1,2,3 --pointMaxW 0.5 --probMargin 0 --t 10

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-1.yaml --gpus 0,1,2,3 --pointMaxW 2 --probMargin 0 --t 10


---

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-1.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 4

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm_sgd_lr-1.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 1

------
------



### pose @ wh (try different temper on baseline p0,m0,rs8421,t10)

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 7

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 13



---

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0.5 --probMargin 0 --t 10


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --rs 4,3,2,1 --t 10


---



-------
-------



### oldpose @ nj (try rs 4,3,2,1 on baseline p0,m0.7,r4321,t1)

ylaunch python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --rs 4,3,2,1 --t 10

ylaunch python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0.7 --rs 4,3,2,1 --t 10

ylaunch python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --rs 4,3,2,1

ylaunch python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 --rs 4,3,2,1

## 03.08


### oldpose @ nj (add pretrain)


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 1 --probMargin .8


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 --t 4

---


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 --t 10


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 4


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin 0 --t 10

---

ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet50/384_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet101/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 


ylaunch --gpu=4 --memory=80000 --cpu=8 -- python pose_estimation/train.py --cfg experiments/mpii/resnet152/256_msssm.yaml --gpus 0,1,2,3 --pointMaxW 0 --probMargin .7 

---------
---------






-------


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
