# pytorch-warmup-cosine-lr

paper : Bag of Tricks for Image Classification with Convolutional Neural Networks (https://arxiv.org/abs/1812.01187)

![Figure_1](https://user-images.githubusercontent.com/33244972/61711191-6bf9b900-ad8e-11e9-85f0-e6c55fbc5bc6.png)


## Usage

python scheduler.py

## Import

~~~
from warmup_scheduler.scheduler import GradualWarmupScheduler

v = torch.zeros(10)
optim = torch.optim.SGD([v], lr=0.01)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100, eta_min=0, last_epoch=-1)
scheduler = GradualWarmupScheduler(optim, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
for epoch in range(1, 100): 
    scheduler.step(epoch)
    
~~~
## note!!!!

**max_epoch = num**

for epoch in range(1, **max_epoch**):

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **max_epoch**, eta_min=0, last_epoch=-1)

**To change the epoch, change all of the highlighted text.**
