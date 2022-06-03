import time
import pandas as pd
import numpy as np
import tensorflow as tf

MSE = tf.keras.losses.MeanSquaredError()
BCE = tf.keras.losses.BinaryCrossentropy()


@tf.function
def compute_pnet_loss(model, x, y):
    # Generate predictions
    z = model.encode(x)
    x_reconstructed = model.decode(z)
    y_pred = model.classify(x)

    # Calculate the classification loss
    class_loss = BCE(y, y_pred)
    # Calculate the reconstruction loss
    recons_loss = MSE(x, x_reconstructed)

    # Calculate total loss
    total = class_loss + 1000*recons_loss

    return total


@tf.function
def compute_bce(model, x, y):
    y_pred = model.classify(x)
    # Calculate the classification loss
    class_loss = BCE(y, y_pred)
    return class_loss


@tf.function
def compute_mse(model, x):
    # Generate predictions
    z = model.encode(x)
    x_reconstructed = model.decode(z)

    # Calculate the reconstruction loss
    recons_loss = MSE(x, x_reconstructed)
    return recons_loss


@tf.function
def train_step(model, x, y, optimizer):

    with tf.GradientTape() as tape:
        # Compute loss
        # loss = compute_pnet_loss(model, x, y)
        classification_loss = compute_bce(model, x, y)
        reconstruction_loss = compute_mse(model, x)
        final_loss = classification_loss + 300*reconstruction_loss

    gradients_of_pnet = tape.gradient(final_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_pnet, model.trainable_variables))
    return classification_loss, reconstruction_loss, final_loss


def train(pnet_model, train_dataset, val_dataset, optimizer, epochs):
    # Losses lists
    train_class_losses = []
    train_reconst_losses = []
    train_total_losses = []
    val_class_losses = []
    val_reconst_losses = []
    val_total_losses = []

    # Training loop
    for epoch in range(epochs):
        start = time.time()
        for x, y in train_dataset:
            classification_loss, reconstruction_loss, final_loss = train_step(pnet_model, x, y, optimizer)
        end_time = time.time()

        # Calculate loss in the validation set
        for sample_val_batch in val_dataset:
            x_val, y_val = sample_val_batch
            val_classification_loss = compute_bce(pnet_model, x_val, y_val)
            val_reconstruction_loss = compute_mse(pnet_model, x_val)
            val_final_loss = val_classification_loss + 300 * val_reconstruction_loss

        print(f'Epoch: {epoch+1}, Classification loss: {classification_loss:.3f}, Val Classification loss: {val_classification_loss:.3f},\n'
              f'Reconstruction loss: {reconstruction_loss:.3f}, Val Reconstruction loss: {val_reconstruction_loss:.3f},\n'
              f'Total loss: {final_loss:.3f}, Val Total loss: {val_final_loss:.3f},\n'
              f'Time elapse for current epoch: {end_time - start:.3f}s\n')

        # Add losses to their lists
        train_class_losses.append(classification_loss.numpy())
        train_reconst_losses.append(reconstruction_loss.numpy())
        train_total_losses.append(final_loss.numpy())
        val_class_losses.append(val_classification_loss.numpy())
        val_reconst_losses.append(val_reconstruction_loss.numpy())
        val_total_losses.append(val_final_loss.numpy())

    # Construct dataframe with losses evolution
    history_df = pd.DataFrame({'train_class_bce': train_class_losses, 'val_class_bce': val_class_losses,
                               'train_recons_mse': train_reconst_losses, 'val_recons_mse': val_reconst_losses,
                               'train_total_loss': train_total_losses, 'val_total_loss': val_total_losses})
    return history_df
