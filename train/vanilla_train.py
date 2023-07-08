from tqdm import tqdm


def vanilla_train_loop(num_epochs, model, optimizer, train_loader, device, kld_weights, reconstruction_sample):
    """
    Runs the training loop for the vanilla model (should not be used for the conditional model)
    :param num_epochs:
    :param model:
    :param optimizer:
    :param train_loader: data loader of the training samples
    :param device: the device to run it (cuda recommended)
    :param kld_weights: the kld_weights
    :param reconstruction_sample: a sample img from the validation set to check how well the model reconstructs images
    :return: the loss per epoch, the reconstructed image, the original reconstruction_sample
    """
    loss_per_epoch = []

    # validation_image, label = val_dataset[0]
    # validation_image = validation_image.unsqueeze(0)
    reconstruction_sample = reconstruction_sample.unsqueeze(0)
    reconstructed_images = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()
        for x, y in tqdm(train_loader):
            # forward pass
            x = x.to(device)
            x_reconst, x, mu, log_var = (*model(x),)

            # loss
            loss = model.loss_function(x_reconst, x, mu, log_var, M_N=kld_weights)['loss']
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = running_loss / len(train_loader)
        print(avg_loss)
        loss_per_epoch.append(avg_loss)

        model.eval()
        reconstruction_sample.to(device)
        x_reconst, x, mu, log_var = model(reconstruction_sample.cuda())
        reconstructed_images.append(x_reconst)

    return loss_per_epoch, reconstructed_images, reconstruction_sample
