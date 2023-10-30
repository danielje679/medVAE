# medVAE

**medVAE** is a project that implements two different Variational Autoencoder (VAE) models: VanillaVAE and ConditionalVAE. The VanillaVAE is a standard VAE, while the ConditionalVAE incorporates a conditional label using data from the [medMNIST](https://medmnist.com/) dataset. This project focuses on medical image datasets, specifically ChestMNIST, OrganMNIST, and OCTMNIST, which share the same number of channels and consist of a substantial number of samples.

## Project Overview
- **Vanilla VAE**: Utilizes the ChestMNIST dataset.
- **Conditional VAE**: Combines data from ChestMNIST, OrganMNIST, and OCTMNIST. It assigns one of three labels for each of the modalities.
- **Architecture**: Both models share a similar architecture.
  - Encoder with 6 Blocks: Conv2d(filter=3x3, channels=[32, 64, 128, 256, 512], stride=2)
  - Batch Normalization
  - Leaky ReLU
  - Decoder with channels in reversed order
  - Input size: 28x28
- **Training Parameters**:
  - Learning Rate: 3e-4
  - Epochs: 50
  - Loss Function: Mean Squared Error (MSE) + Kullback-Leibler Divergence (KLD)
- **Dataset Information**:
  - Vanilla VAE trained on ChestMNIST with 78,468 samples.
  - Conditional VAE trained on all three datasets with a total of 210,526 samples.
  - Both models trained for 50 epochs each.

## Project Structure
The code is organized as follows:

- `data`: Contains code related to data preprocessing.
- `evaluation`: Includes scripts for model evaluation.
- `model_checkpoints`: Stores model checkpoints.
- `models`: model definitions for VanillaVAE and ConditionalVAE.
- `train`: Contains scripts for training the models.
- `requirements.txt`: Lists project dependencies.
- `train_and_evaluate.ipynb`: A Jupyter Notebook that provides a guide to training and evaluating the models.

## Getting Started
1. Clone the repository to your local machine.
2. Install the required dependencies by running: `pip install -r requirements.txt`.
3. Use `main.py` or follow the instructions in `train_and_evaluate.ipynb` to train and evaluate the models.
4. You can also explore different datasets, training configurations, and evaluation methods by modifying the provided code.

## Acknowledgments
- The medVAE project was developed as part of the course *Deep Generative models*, at TU Darmstadt.

## Contact
For any questions or inquiries regarding this project, please contact uni@danieljeckel.de or create an issue.
