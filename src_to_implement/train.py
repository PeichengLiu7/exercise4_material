import torch as t
# import torch.load as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import time

# hyperparameters
NUM_EPOCHES = 30
BATCH_SIZE = 40
LEARNING_RATE = 1e-4
EARLYSTOP_PATIENCE = 8


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
################################################################
csv_file_path="data.csv"
data_pd=pd.read_csv(csv_file_path,sep=";")
# 再利用model_selection模块将整个data_pd划分为对应的数据集和验证集，如果没有test_zie参数的话，
# 默认是数据集的25%作为validation_set,
train_pd,val_pd=train_test_split(data_pd,test_size=0.25)
print(f"number of elements in train_pd:{train_pd.shape[0]}")
print(f"number of elements in val_pd:{val_pd.shape[0]}")
print(data_pd.head())



# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
# 然后根据上面划分的train_pd, val_pd来实例化ChallengeDataset类并以此建立DataLoader实例，
# 并分别划分不同的模式"train"和"val"
train_set=t.utils.data.DataLoader(ChallengeDataset(train_pd,mode="train"),
    batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
val_set=t.utils.data.DataLoader(ChallengeDataset(val_pd,mode="val"),
    batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

# 直接求取train_set的长度，表示的是一个epoch中含有多少个batch，此处drop_last=False，
# 每个batch大小为BATCH_SIZE=100，train_pd数据长度为1500，因此len(train_set)=15
print(f"length of train_set:{len(train_set)}")
print(f"length of val_set:{len(val_set)}")



# create an instance of our ResNet model
# TODO
resnet=model.ResNet()



# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
loss_criterion=t.nn.BCELoss()
optimizer=t.optim.Adam(resnet.parameters(),lr=LEARNING_RATE)

trainer=Trainer(model=resnet,crit=loss_criterion,optim=optimizer,
                train_dl=train_set,val_test_dl=val_set,cuda=False,
                early_stopping_patience=EARLYSTOP_PATIENCE)

start=time.time()
# go, go, go... call fit on trainer
#TODO
# trainer.restore_checkpoint(2)
res=trainer.fit(NUM_EPOCHES)

elapsed=time.time()-start
print("training time:{:02d}min{:02d}s".format(int(elapsed//60),int(elapsed%60)))

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
# 将当前的时间作为保存的文件名
current_time=datetime.datetime.now()
pic_time="loss_{}_{}_{}_{}_{}_{}.png".format(current_time.year,
         current_time.month,current_time.day,current_time.hour,
         current_time.minute,current_time.second)
plt.savefig("loss.png",bbox_inches="tight")
plt.show()
