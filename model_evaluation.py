"""
model_evaluation.py
Handles plotting of training and testing curves.
"""
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_curves(train_loss, test_loss, train_acc, test_acc, folder_name='plots'):
    """
    Plots the Loss and Accuracy curves for Training and Testing.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    epochs = range(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, test_loss, label='Test Loss', color='red', linestyle='--')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, 'loss_curve.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='green')
    plt.plot(epochs, test_acc, label='Test Accuracy', color='orange', linestyle='--')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, 'accuracy_curve.png'))
    plt.close()
    
    print(f"Curves saved to {folder_name}/ directory.")