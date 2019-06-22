from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        layers = []
        in_channels, in_h, in_w, = tuple(self.in_size)
        filters = [128, 256, 512, 1024]
        stride = 2
        for current_filter in filters:
            layers.append(nn.Conv2d(in_channels, current_filter, kernel_size=5, padding=stride, stride=stride))
            layers.append(nn.BatchNorm2d(current_filter))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = current_filter

        self.feature_extractor = nn.Sequential(*layers)

        in_features = int(filters[-1] * (in_h // (stride ** len(filters))) * (in_w // (stride ** len(filters))))
        self.classifier = nn.Sequential(nn.Linear(in_features, 1))
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.feature_extractor.forward(x)
        #import ipdb;ipdb.set_trace()
        y = y.reshape(y.shape[0], -1)  # flatten output 4D tensor to 1 column vector
        y = self.classifier.forward(y)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======
        hidden_dims = [1024, 512, 256, 128]
        self.first_feature_shape = (hidden_dims[0], featuremap_size, featuremap_size)
        in_channels = self.first_feature_shape[0]
        self.expander = nn.Linear(z_dim, int(torch.prod(torch.Tensor(self.first_feature_shape)).item()))

        layers = []
        stride = 2
        for current_dim in hidden_dims[1:]:
            layers.append(nn.ConvTranspose2d(in_channels, current_dim, kernel_size=5, stride=stride, padding=stride, output_padding=1))
            layers.append(nn.BatchNorm2d(current_dim))
            layers.append(nn.ReLU())
            in_channels = current_dim

        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=stride, output_padding=1))
        layers.append(nn.Tanh())
        self.feature_decoder = nn.Sequential(*layers)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        import contextlib
        with torch.no_grad() if not with_grad else contextlib.nullcontext():
            z = torch.randn((n, self.z_dim), device=device, requires_grad=with_grad)
            samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        x = self.expander(z)
        x = x.reshape((-1, *self.first_feature_shape))
        x = self.feature_decoder(x)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss_function = nn.BCEWithLogitsLoss()
    dist = torch.distributions.uniform.Uniform(-label_noise / 2, label_noise / 2)
    y_data_target = dist.sample(y_data.shape) + data_label
    y_generated_target = dist.sample(y_generated.shape) + (1 - data_label)

    loss_data = loss_function(y_data, y_data_target)
    loss_generated = loss_function(y_generated, y_generated_target)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_function = nn.BCEWithLogitsLoss()
    y_generated_target = torch.zeros(y_generated.shape) + data_label
    loss = loss_function(y_generated, y_generated_target)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()

    N = len(x_data)
    x_generated = gen_model.sample(N, True)

    x_data_scores = dsc_model.forward(x_data)
    x_generated_scores = dsc_model.forward(x_generated.detach())

    dsc_loss = dsc_loss_fn(x_data_scores, x_generated_scores)
    dsc_loss.backward()

    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    x_generated_scores = dsc_model.forward(x_generated)

    gen_loss = gen_loss_fn(x_generated_scores)
    gen_loss.backward()

    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()
