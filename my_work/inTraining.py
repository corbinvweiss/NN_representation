#inTrainingGrid.py
# show the PCA visualizations of the MLP for all epochs side by side.

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data from the MLP in training into memory
with open("inTraining.json", "r") as file:
    raw_data = json.load(file)

data = raw_data

# Get the mean values of classification certainties for each class across all epochs
epoch_mean_classes_values = []
for epoch in range(len(data)):
    mean_classes_values = []
    for i in range(10):
        input_class = data[epoch][i]
        mean_certainties = []
        for j in range(10):
            class_certainties = [sample[j] for sample in input_class]
            mean_certainties.append(np.mean(class_certainties))
        mean_classes_values.append(mean_certainties)
    epoch_mean_classes_values.append(mean_classes_values)

# Define the distance between classes to be 1 - certainty
epoch_distances = [[[np.amax(class_vals) - val for val in class_vals] 
                    for class_vals in epoch_mean_classes_values[epoch]] 
                    for epoch in range(len(epoch_mean_classes_values))]

# Store the PCA data for each epoch
epoch_PCAs = []
for distances in epoch_distances:
    D = np.array(distances)
    D_sym = (D + D.T) / 2
    X = D_sym
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    epoch_PCAs.append((X_pca, pca.explained_variance_ratio_))

def show_all_figures():
    """
    Display all PCA visualizations in a grid with 2 rows and 5 columns.
    Each visualization is labeled with the epoch number and the explained variance ratio.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        X_pca, explained_variance = epoch_PCAs[i]
        for j in range(X_pca.shape[0]):
            ax.scatter(X_pca[j, 0], X_pca[j, 1], label=f"Digit {j}")
            ax.text(X_pca[j, 0] + 0.005, X_pca[j, 1] + 0.005, str(j), fontsize=8)
        ax.set_title(f"Epoch {i+1}")
        # ax.set_xlabel("Principal Component 1")
        # ax.set_ylabel("Principal Component 2")
        ax.grid(True)
        # ax.legend()
        ax.text(0.5, -0.1, f"Explained Variance: [{explained_variance[0]:.2f}, {explained_variance[1]:.2f}]", 
                size=10, ha="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

def show_single_figure(epoch_index):
    """
    Display the PCA visualization for a single epoch.
    epoch_index: Index of the epoch to visualize.
    """
    if epoch_index < 0 or epoch_index >= len(epoch_PCAs):
        print("Invalid epoch index")
        return

    X_pca, explained_variance = epoch_PCAs[epoch_index]

    plt.figure(figsize=(8, 6))
    for i in range(X_pca.shape[0]):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], label=f"Digit {i}")
        plt.text(X_pca[i, 0] + 0.005, X_pca[i, 1] + 0.005, str(i), fontsize=12)
    
    plt.title(f"PCA Visualization of Epoch {epoch_index+1}")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    plt.grid(True)
    # plt.legend()
    plt.text(0.5, -0.1, f"Explained Variance: [{explained_variance[0]:.2f}, {explained_variance[1]:.2f}]", 
             size=10, ha="center", transform=plt.gca().transAxes)
    plt.show()

if __name__ == "__main__":
    # show_all_figures()
    show_single_figure(9)  # Visualize the PCA for the given epoch
