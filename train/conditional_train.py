from tqdm import tqdm
import torch


def cond_train_loop(num_epochs, model, optimizer, train_loader, device, kld_weights, reconstruction_sample):
    loss_per_epoch = []

    # validation_image, label = val_dataset[0]
    # validation_image = validation_image.unsqueeze(0)
    # reconstructed_images = []

    reconstructed_images = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for x, y in tqdm(train_loader):
            # forward pass
            x = x.cuda()

            y = torch.eye(3)[y].squeeze().to(device)
            x_reconst, input, mu, log_var = (*model(x, y),)

            # loss
            loss = model.loss_function(x_reconst, input, mu, log_var, M_N=kld_weights)['loss']
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = running_loss / len(train_loader)
        print(avg_loss)
        loss_per_epoch.append(avg_loss)

        model.eval()

    return loss_per_epoch, reconstructed_images, reconstruction_sample