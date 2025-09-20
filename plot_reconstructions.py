import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os

def plot_reconstructions_vs_originals():
    """
    Plot the first 10 reconstructed faces from ./data/reconstructed_faces/50/ 
    against the first 10 faces in the testing sample
    """
    
    # Paths
    reconstructed_dir = "./data/reconstructed_faces/50/"
    testing_dir = "./data/testing_hsv/"
    
    # Check if directories exist
    if not os.path.exists(reconstructed_dir):
        print(f"Error: {reconstructed_dir} does not exist!")
        return
    
    if not os.path.exists(testing_dir):
        print(f"Error: {testing_dir} does not exist!")
        return
    
    # Get the first 10 reconstructed faces
    reconstructed_files = []
    for i in range(1, 11):  # First 10 faces
        filename = f"face_{800+i}_reconstructed.jpg"
        filepath = os.path.join(reconstructed_dir, filename)
        if os.path.exists(filepath):
            reconstructed_files.append(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    # Get the first 10 testing faces
    testing_files = []
    for i in range(1, 11):  # First 10 faces
        filename = f"{800+i}.jpg"
        filepath = os.path.join(testing_dir, filename)
        if os.path.exists(filepath):
            testing_files.append(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    if len(reconstructed_files) == 0 or len(testing_files) == 0:
        print("No files found to plot!")
        return
    
    # Load images
    reconstructed_images = []
    testing_images = []
    
    for filepath in reconstructed_files:
        try:
            img = skimage.io.imread(filepath)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Convert RGB to grayscale
                img = np.mean(img, axis=2).astype(np.uint8)
            reconstructed_images.append(img)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    for filepath in testing_files:
        try:
            img = skimage.io.imread(filepath)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Convert RGB to grayscale
                img = np.mean(img, axis=2).astype(np.uint8)
            testing_images.append(img)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if len(reconstructed_images) == 0 or len(testing_images) == 0:
        print("No images loaded successfully!")
        return
    
    # Create the plot
    n_images = min(len(reconstructed_images), len(testing_images), 10)
    
    fig, axes = plt.subplots(2, n_images, figsize=(2*n_images, 4))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    # Plot original faces (top row)
    for i in range(n_images):
        axes[0, i].imshow(testing_images[i], cmap='gray')
        axes[0, i].set_title(f'Original {801+i}', fontsize=10)
        axes[0, i].axis('off')
    
    # Plot reconstructed faces (bottom row)
    for i in range(n_images):
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {801+i}', fontsize=10)
        axes[1, i].axis('off')
    
    # Add overall title
    fig.suptitle('Original vs Reconstructed Faces\n(50 Eigenfaces)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./data/reconstruction_comparison_50_eigenfaces.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and display reconstruction errors
    print(f"\nReconstruction Analysis (50 Eigenfaces):")
    print(f"Number of face pairs compared: {n_images}")
    
    total_error = 0
    for i in range(n_images):
        # Calculate MSE between original and reconstructed
        mse = np.mean((testing_images[i].astype(float) - reconstructed_images[i].astype(float))**2)
        total_error += mse
        print(f"Face {801+i}: MSE = {mse:.2f}")
    
    avg_error = total_error / n_images
    print(f"Average MSE: {avg_error:.2f}")
    
    # Save error statistics
    with open('./data/reconstruction_error_50_eigenfaces.txt', 'w') as f:
        f.write(f"Reconstruction Error Analysis (50 Eigenfaces)\n")
        f.write(f"Number of face pairs compared: {n_images}\n")
        f.write(f"Average MSE: {avg_error:.2f}\n\n")
        f.write("Individual face errors:\n")
        for i in range(n_images):
            mse = np.mean((testing_images[i].astype(float) - reconstructed_images[i].astype(float))**2)
            f.write(f"Face {801+i}: MSE = {mse:.2f}\n")

if __name__ == "__main__":
    print("Plotting reconstructed faces vs original testing faces...")
    plot_reconstructions_vs_originals()
    print("Done!") 