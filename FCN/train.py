import os
import torch
from torch.utils.data import DataLoader

from networks import VGGNet, FCN8s
from citydataset import city,classes_city,cityscapes_classes, maps_classes, facades_classes
from loss import cross_entropy2d, CrossEntropyLoss2d
from utils import Train

########################################################################################################################
batch_size = 16
epochs = 500
lr = 1e-4
momentum = 0
w_decay = 1e-5
step_size = 50
gamma = 0.5


USE_CUDA = True
NUM_CLASSES = 31
CHECKPOINTS_FOLDER = r"./checkpoints/"
EXP = "EXP_" + str(len(os.listdir(CHECKPOINTS_FOLDER))+1)
SAVE_FOLDER = os.path.join(CHECKPOINTS_FOLDER, EXP)
os.mkdir(SAVE_FOLDER)
print("SAVING RESULTS IN : " + SAVE_FOLDER)
########################################################################################################################


########################################################################################################################
##### Modele
vgg_model = VGGNet(requires_grad=True)
fcn = FCN8s(pretrained_net=vgg_model, n_class=len(facades_classes)).cuda()


##### Optimizer and Scheduler
optimizer = torch.optim.RMSprop(fcn.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

##### Loss
loss = CrossEntropyLoss2d()
##### Dataset and Dataloader
city_train = city(mode="train",classes=facades_classes)
city_train_loader  = DataLoader(dataset=city_train, batch_size=batch_size)

city_val = city(mode="val", classes=facades_classes)
city_val_loader = DataLoader(dataset=city_val, batch_size=batch_size)

########################################################################################################################

Train(
    model=fcn,
    optimizer=optimizer,
    scheduler=scheduler,
    loss=loss,
    train_loader=city_train_loader,
    val_loader=city_val_loader,
    epochs=epochs,
    save_folder=SAVE_FOLDER,
    use_cuda=USE_CUDA
)
########################################################################################################################
