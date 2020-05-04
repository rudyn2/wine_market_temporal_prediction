import numpy as np
import torch


def train(model, optimizer, loss_fn, train_loader, valid_loader, epochs: int = 10, verbose: bool = True):
    mean_train_losses = []
    mean_valid_losses = []

    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        # train step
        for i, (input_seq, output_seq) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(input_seq)
            loss = loss_fn(outputs, output_seq)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # calculates validation loss
        model.eval()
        with torch.no_grad():
            for i, (input_seq, output_seq) in enumerate(valid_loader):
                outputs = model(input_seq)
                loss = loss_fn(outputs, output_seq)
                valid_losses.append(loss.item())

        # saves losses
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))

        # prints progress
        if verbose:
            print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}' \
                  .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses)))
    return mean_train_losses, mean_valid_losses
