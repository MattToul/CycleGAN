import os
import numpy as np
import torch
import pickle

def train_epoch(model, dataloader, epoch, optimizer, loss,use_cuda=True):
    applyCuda = lambda x: x.cuda() if use_cuda else x
    losses = []
    model.train()
    for idx, batch in enumerate(dataloader):
        images = applyCuda(batch['image'])
        targets = applyCuda(batch['target'])

        predictions = model(images)
        try:
            Loss = loss(predictions, targets)
        except:
            print(idx)
            print(images.size())
            print(targets.size())
            print(predictions.size())
            raise ()
        model.zero_grad()
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        losses.append(Loss.data.cpu().numpy())

        if idx%80==0:
            print("Epoch {} - batch {} - Loss {}".format(epoch, idx, Loss.data.cpu().numpy()))
    return np.mean(losses)

def validate(model, dataloader,loss,use_cuda=True):
    applyCuda = lambda x: x.cuda() if use_cuda else x
    losses = []
    model.eval()
    for idx, batch in enumerate(dataloader):
        images = applyCuda(batch['image'])
        targets = applyCuda(batch['target'])

        predictions = model(images)
        Loss = loss(predictions, targets)
        losses.append(Loss.data.cpu().numpy())
    return np.mean(losses)

def Train(
        model,
        optimizer,
        scheduler,
        loss,
        train_loader,
        val_loader,
        epochs,
        save_folder,
        use_cuda):
    save_checkpoint = os.path.join(save_folder,'checkpoint.pth.tar')
    train_losses = []
    val_losses = []

    print("STARTING TRAINING")
    for epoch in range(epochs):
        scheduler.step()
        train_loss = train_epoch(model, dataloader=train_loader, epoch=epoch, optimizer=optimizer, loss=loss, use_cuda=use_cuda)
        val_loss = validate(model, dataloader=val_loader, loss=loss, use_cuda=use_cuda)
        print("EPOCH " + str(epoch) + "/"+str(epochs))
        print("TRAINING LOSS   "+ str(train_loss))
        print("VALIDATION LOSS "+str(val_loss))
        print("--------------------------------------------------")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss == np.min(val_losses):
            torch.save(model.state_dict(), save_checkpoint)


    print("BEST MODEL AT EPOCH : " + str(np.argmin(val_losses)))

    with open(os.path.join(save_folder, "train_losses.pickle"), 'wb') as handle:
        pickle.dump(train_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_folder, "val_losses.pickle"), 'wb') as handle:
        pickle.dump(val_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
