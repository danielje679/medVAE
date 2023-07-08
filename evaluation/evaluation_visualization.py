import matplotlib.pyplot as plt

def plot_image_grid(original_image, reconstructed_images):
    num_images = len(reconstructed_images)
    num_plotted_imgs = 5
    step_size = int(num_images/num_plotted_imgs)
    fig, axes = plt.subplots(2, num_plotted_imgs, figsize=(10, 4))

    original_image_reshaped = original_image[0].permute(1, 2, 0)

    # Plot original image multiple times on the top row
    for i in range(num_plotted_imgs):
        axes[0, i].imshow(original_image_reshaped, cmap='gray')
        axes[0, i].set_axis_off()

    # Plot reconstructed images on the bottom row
    for i in range(0, num_images, step_size):

        reconstructed_image = reconstructed_images[i].cpu()[0]
        reconstructed_image = reconstructed_image.permute(1, 2, 0)
        reconstructed_image = reconstructed_image.detach().numpy()
        axes[1, int(i/step_size)].imshow(reconstructed_image, cmap='gray')
        axes[1, int(i/step_size)].set_xlabel(f'epoch {i}')
        axes[1, int(i/step_size)].get_xaxis().set_ticks([])#set_axis_off()
        axes[1, int(i/step_size)].get_yaxis().set_ticks([])


    plt.tight_layout()
    plt.show()