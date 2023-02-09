import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import f1_score


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=False,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        # 自适应调整学习率
        if self._optim:
            self._lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self._optim, mode="min", patience=40, factor=0.5, verbose=True)
            # self._lr_scheduler=t.optim.lr_scheduler.CosineAnnealingLR(optimizer=self._optim,T_max=3)

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called.
        # This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # TODO
        # 单步训练的意思？
        # assume that x in the input_tensor while y is corresponding labels

        # 首先将input_tensor输入模型，然后计算loss
        outputs = self._model(x)
        # 因为输出是sigmoid，而实际上labels只有四种类型[0,0],[0,1],[1,0],[1,1]，
        # 因此让outputs中>=mean(oouputs)的归为1，小于均值的的归为0

        loss = self._crit(outputs, y)
        # 因为梯度累加的原因，所以在开始反向传播之前需要将梯度清零
        self._optim.zero_grad()
        loss.backward()

        self._optim.step()
        # 新增
        self._lr_scheduler.step(loss)

        return loss

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO

        predictions = self._model(x)

        loss = self._crit(predictions, y)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        return predictions, loss

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # TODO
        # 从train.py文件的训练流程来看已经经过了包装，也就是传进来的training_set已经是DataLoader的一个实例
        # 所以应该是让完成如何在一个epoch下进行训练
        # 用来存储每经过一次batch所产生的loss，然后把batch_num次产生的loss平均就是当前一个epoch的平均loss
        loss = np.zeros(len(self._train_dl))
        for idx, (x, y) in enumerate(self._train_dl):
            # dataloader格式的变量，自动给你取batch_size个数据，没有设置shuffle, train_pd[60],
            # 迭代遍历载入进来的训练数据集
            # idx表示第几次batch
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            loss[idx] = self.train_step(x, y)

        return np.average(loss)

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice.
        # You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # TODO
        # 首先将模型设置为eval()模式，难点是如何评价模型的性能标准，因为是多分类问题，并且最后输出是sigmoid函数，
        # sigmoid函数的值域连续分布在(0,1)之间，因此最后的prediction应该是两个形如[0.2,0.8],[0.1,0.4]这样的值，
        # 在这种情况下，判定标准是什么？（因为y值是[0,1],[1,0],[1,1],[0,0]四个离散值，该采用何种标准来测试accuracy呢？计算loss
        # 可以使用交叉熵损失。在此设想一个简单的判别标准，大于0.5的一律归为1，小于一律归为0
        self._model.eval()
        # 将模型的梯度计算取消，因为是test阶段
        with t.no_grad():
            # 存储val_test阶段产生的loss，用于计算平均loss
            loss = np.zeros(len(self._val_test_dl))
            # 预测正确的样本数
            accurate = 0
            # 参与预测的总样本数
            total = 0

            # add-on
            y_pred = []
            y_true = []

            for idx, [x, y] in enumerate(self._val_test_dl):
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                predicts, loss[idx] = self.val_test_step(x, y)
                # 然后将predicts_trans与y作比较，计算准确率
                # 累计总共参加val的样本数
                total += y.size(0)  # (batch_size,-1)
                # 将其中预测准确的数量加起来
                accurate += sum(row.all().int().item() for row in predicts == y)

                y_pred.append(predicts.cpu().numpy().flatten())
                y_true.append(y.cpu().numpy().flatten())

            print(f"The Average metrics:{accurate * 100.0 / total}%", end=",")
            f1scores = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
            print("f1 score:{:.4f}".format(f1scores), end=",")
        return np.average(loss), f1scores

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO
        train_loss = []
        val_loss = []
        epoch_counter = 0
        # 衡量val_loss停止更新的标准
        tolerance_error = 1e-5
        # 早停计数器，达到early_stopping_patience即退出循环
        earlystop_counter = 0
        # 训练过程中最佳的损失
        best_val_loss = 0
        # 保存最佳损失的状态量
        best_model_state = None
        # 保存最佳损失出现的epoch序号，也就是val_loss最小的时候是哪一次epoch
        best_epoch_num = 0
        # 保存最高的f1_score
        best_f1_score = 0

        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            # TODO

            train_loss.append(self.train_epoch())
            val_loss_per_epoch, f1_score_per_epoch = self.val_test()
            val_loss.append(val_loss_per_epoch)
            epoch_counter += 1
            print("Epoch:{:03d},train_loss:{:.6f},val_loss:{:.6f}" \
                  .format(epoch_counter, train_loss[-1], val_loss[-1]))
            if len(val_loss) <= 1:
                best_val_loss = val_loss[0]
                best_model_state = self._model.state_dict()
                best_epoch_num = epoch_counter
                best_f1_score = f1_score_per_epoch

            if best_val_loss - val_loss[-1] > tolerance_error:
                best_val_loss = val_loss[-1]
                earlystop_counter = 0
            else:
                earlystop_counter += 1

            # 以更高的f1_score为标准，选定model参数
            if best_f1_score < f1_score_per_epoch:
                best_f1_score = f1_score_per_epoch
                best_epoch_num = epoch_counter
                best_model_state = self._model.state_dict()

            if earlystop_counter >= self._early_stopping_patience or epoch_counter >= epochs:
                # 将模型恢复到最好的状态，并在此时保存checkpoint
                self._model.load_state_dict(best_model_state)
                self.save_checkpoint(best_epoch_num)
                print(f"\nThe best epoch is {best_epoch_num} with f1 score:{best_f1_score}")
                break

        return train_loss, val_loss









