import time
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import pairwise_dist
import config

MSE = tf.keras.losses.MeanSquaredError()
CCE = tf.keras.losses.CategoricalCrossentropy()


@tf.function
def compute_pnet_loss(model, x, y):
    # Generate predictions
    z = model.encode(x)
    x_reconstructed = model.decode(z)
    y_pred = model.classify(x)

    # Calculate the classification loss
    class_loss = CCE(y, y_pred)
    # Calculate the reconstruction loss
    recons_loss = MSE(x, x_reconstructed)

    # Calculate total loss
    total = class_loss + 1000*recons_loss

    return total


@tf.function
def compute_bce(model, x, y):
    # Generate predictions
    y_pred = model.classify(x)
    # Calculate the classification loss
    class_loss = CCE(y, y_pred)
    return class_loss


@tf.function
def compute_mse(model, x):
    # Generate codes
    z = model.encode(x)
    # Reconstruct
    x_reconstructed = model.decode(z)
    # Calculate the reconstruction loss
    recons_loss = MSE(x, x_reconstructed)
    return recons_loss


@tf.function
def compute_reg1(model, x):
    # Generate codes
    z = model.encode(x)
    # Get prototypes
    prototypes = model.get_prototypes()[0]
    # Calculate pairwise distances
    feature_vector_distances = pairwise_dist(prototypes, z)
    return tf.reduce_mean(tf.reduce_min(feature_vector_distances, axis=1))


@tf.function
def compute_reg2(model, x):
    # Generate codes
    z = model.encode(x)
    # Get prototypes
    prototypes = model.get_prototypes()[0]
    # Calculate pairwise distances
    prototype_distances = pairwise_dist(z, prototypes)
    return tf.reduce_mean(tf.reduce_min(prototype_distances, axis=1), name='error_2')


@tf.function
def train_step(model, x, y, optimizer):

    with tf.GradientTape() as tape:
        # Compute loss
        # loss = compute_pnet_loss(model, x, y)
        classification_loss = compute_bce(model, x, y)
        reconstruction_loss = compute_mse(model, x)
        reg1_loss = compute_reg1(model, x)
        reg2_loss = compute_reg2(model, x)
        final_loss = classification_loss + config.l_recons*reconstruction_loss + config.l_reg1*reg1_loss +\
                     config.l_reg2*reg2_loss

    gradients_of_pnet = tape.gradient(final_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_pnet, model.trainable_variables))
    return classification_loss, reconstruction_loss, reg1_loss, reg2_loss, final_loss


def train(pnet_model, train_dataset, val_dataset, optimizer, epochs, checkpoints_epoch=None):
    # Losses lists
    train_class_losses = []
    train_reconst_losses = []
    train_total_losses = []
    val_class_losses = []
    val_reconst_losses = []
    val_total_losses = []
    train_reg1_losses = []
    val_reg1_losses = []
    train_reg2_losses = []
    val_reg2_losses = []
    prototype_checkpoints = []

    # Training loop
    for epoch in range(epochs):
        start = time.time()
        for x, y in train_dataset:
            classification_loss, reconstruction_loss, reg1_loss, reg2_loss, final_loss = train_step(pnet_model,
                                                                                                    x, y, optimizer)
        end_time = time.time()

        # Calculate loss in the validation set
        for sample_val_batch in val_dataset:
            x_val, y_val = sample_val_batch
            val_classification_loss = compute_bce(pnet_model, x_val, y_val)
            val_reconstruction_loss = compute_mse(pnet_model, x_val)
            val_reg1_loss = compute_reg1(pnet_model, x_val)
            val_reg2_loss = compute_reg2(pnet_model, x_val)
            val_final_loss = val_classification_loss + config.l_recons * val_reconstruction_loss + \
                                config.l_reg1 * val_reg1_loss + config.l_reg2 * val_reg2_loss

        print(f'Epoch: {epoch+1}, Classification loss: {classification_loss:.3f}, Val Classification loss: {val_classification_loss:.3f},\n'
              f'Reconstruction loss: {reconstruction_loss:.3f}, Val Reconstruction loss: {val_reconstruction_loss:.3f},\n'
              f'Total loss: {final_loss:.3f}, Val Total loss: {val_final_loss:.3f},\n'
              f'Time elapse for current epoch: {end_time - start:.3f}s\n')

        # Add losses to their lists
        train_class_losses.append(classification_loss.numpy())
        train_reconst_losses.append(reconstruction_loss.numpy())
        train_reg1_losses.append(reg1_loss)
        train_reg2_losses.append(reg2_loss)
        train_total_losses.append(final_loss.numpy())
        val_class_losses.append(val_classification_loss.numpy())
        val_reconst_losses.append(val_reconstruction_loss.numpy())
        val_reg1_losses.append(val_reg1_loss.numpy())
        val_reg2_losses.append(val_reg2_loss.numpy())
        val_total_losses.append(val_final_loss.numpy())

        # Save prototypes checkpoint
        if checkpoints_epoch is not None:
            if (epoch + 1) % checkpoints_epoch == 0:
                prototype_checkpoint = pnet_model.pnet.get_layer('proto_layer').get_weights()
                prototype_checkpoints.append(prototype_checkpoint[0])

    # Construct dataframe with losses evolution
    history_df = pd.DataFrame({'train_class_bce': train_class_losses, 'val_class_bce': val_class_losses,
                               'train_recons_mse': train_reconst_losses, 'val_recons_mse': val_reconst_losses,
                               'train_reg1_loss': train_reg1_losses, 'val_reg1_loss': val_reg1_losses,
                               'train_reg2_loss': train_reg2_losses, 'val_reg2_loss': val_reg2_losses,
                               'train_total_loss': train_total_losses, 'val_total_loss': val_total_losses})
    return history_df, prototype_checkpoints


def train_prototype_network(pnet_model, train_dataset, val_dataset, optimizer, epochs_1, epochs_2,
                            checkpoints_epoch=None):

    # TRAIN ALL MODEL EXCEPT LAST LAYER
    # Freeze last dense layer for classification
    pnet_model.trainable = True
    for layer in pnet_model.pnet.layers[-2:]:
        layer.trainable = False
    # Train
    history_df1, prototype_checkpoints1 = train(pnet_model, train_dataset, val_dataset, optimizer, epochs_1,
                                                checkpoints_epoch)

    # TRAIN THE DENSE LAYER
    # Froze all model except for the last Dense layer for classification
    pnet_model.encoder.trainable = False
    pnet_model.decoder.trainable = False
    pnet_model.pnet.trainable = True
    for layer in pnet_model.pnet.layers[:-2]:
        layer.trainable = False
    assert pnet_model.encoder.trainable == False
    assert pnet_model.decoder.trainable == False

    # Train
    history_df2, prototype_checkpoints2 = train(pnet_model, train_dataset, val_dataset, optimizer, epochs_2,
                                                checkpoints_epoch)

    # Concat history dfs and checkpoints
    final_hitory_df = pd.concat([history_df1, history_df2], ignore_index=True)
    prototype_checkpoints = prototype_checkpoints1 + prototype_checkpoints2

    return final_hitory_df, prototype_checkpoints
