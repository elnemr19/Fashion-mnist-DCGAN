# Fashion MNIST GAN Project


![image](https://github.com/user-attachments/assets/e4f245d9-2768-4b37-818b-fa96d894772b)


This project uses a **Wasserstein GAN** with **Gradient Penalty (WGAN-GP)** to generate synthetic grayscale images of clothing items, based on the **Fashion MNIST dataset**.





## Table of Contents

1. [Project Overview](https://github.com/elnemr19/Fashion-mnist-DCGAN/blob/main/README.md#project-overview)

2. [Dataset](https://github.com/elnemr19/Fashion-mnist-DCGAN/blob/main/README.md#dataset)

3. [Model Architecture](https://github.com/elnemr19/Fashion-mnist-DCGAN/blob/main/README.md#model-architecture)

4. [Loss Functions](https://github.com/elnemr19/Fashion-mnist-DCGAN/blob/main/README.md#loss-functions)

5. [Training Details](https://github.com/elnemr19/Fashion-mnist-DCGAN/blob/main/README.md#training-details)

6. [Results](https://github.com/elnemr19/Fashion-mnist-DCGAN/blob/main/README.md#results)




## Project Overview

Generative Adversarial Networks (GANs) consist of two models:

1. **Generator:** Creates synthetic images to mimic the dataset.

2. **Discriminator:** Differentiates between real and synthetic images.

This implementation of WGAN-GP stabilizes training by enforcing Lipschitz continuity using gradient penalties. The primary goal is to generate realistic clothing item images similar to those in the Fashion MNIST dataset.




## Dataset

The Fashion MNIST dataset contains grayscale images of 10 clothing categories, each image sized at 28x28 pixels.

**Preprocessing Steps:**

* Normalized pixel values to the range `[-1, 1]` using: `X_train = (X_train - 127.5) / 127.5`


* Reshaped images to include a single-channel dimension: `(28, 28, 1)`.

* Reduced the training set size to half using random sampling for faster training.




## Model Architecture


**Generator**

The generator creates synthetic images from random noise. It uses:

* **Transposed convolutions** for upsampling the feature map and increasing resolution.

* **Batch normalization** to stabilize training and improve convergence.

* **LeakyReLU activations** for non-linearity.

* **Tanh activation** for the output layer, scaling pixel values to the range ` [-1, 1] `.

**Discriminator**

The discriminator classifies images as real or fake. It uses:

* **Convolutional layers:** Extracts features from input images.

* **LeakyReLU activation:** Prevents vanishing gradients.

* **Dropout:** Regularizes the model to reduce overfitting.

* **Fully connected layer:** Outputs a single value representing the real/fake probability.



## Loss Functions



**Generator Loss**

Encourages the generator to produce images that the discriminator classifies as real:


**Discriminator Loss**

Penalizes incorrect classifications and incorporates gradient penalty:


**Gradient Penalty (GP)**

Enforces Lipschitz continuity to stabilize training:



## Training Details


* **Noise Dimension:** 100

* **Epochs:** 200

* **Batch Size:** 128

* **Image Size:** 28x28

* **Optimizers:** Adam with learning rate `1e-4`, beta_1 `0.5`, beta_2 `0.9`.



Training loop includes:

1. Sampling random noise for the generator.

2. Calculating losses for the generator and discriminator.

3. Updating weights using backpropagation.

## Results

After 200 epochs, the generator produces realistic clothing item images resembling those in the Fashion MNIST dataset. Below are some generated samples:

![image](https://github.com/user-attachments/assets/f5d26dd3-eaac-4a7c-b0e3-314f4b0aebe4)










