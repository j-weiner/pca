import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import datetime
import imageio
import skimage
from scipy.io import loadmat
import os
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

#convert_to_hsv()

meanFaceMatrix = np.zeros((800,16384))

for i in range(1,800):
        imgVector = skimage.io.imread("./data/training_hsv/"+f"{i}"+".jpg").flatten()
        meanFaceMatrix[i-1,:] = imgVector #populating the meanFaceMatrix with flattened images
        

def draw_mean_face():
    mean_face = np.mean(meanFaceMatrix, axis=0).reshape(128, 128)
    mean_face_uint8 = mean_face.astype(np.uint8)
    skimage.io.imsave("./data/meanFace.jpg", mean_face_uint8)

#draw_mean_face()

centeredMatrix = np.zeros((800,16384))

for i in range(1,800):
    centeredMatrix[i-1,:] = meanFaceMatrix[i-1,:] - skimage.io.imread("./data/meanFace.jpg").flatten() #populating the centered matrix by subtracing mean face from each image(?)
    
covarianceMatrix = centeredMatrix.T @ centeredMatrix #multiply the separate centered by itself transformed to get the covarianceMatrix

eigenvalues, eigenvectors = np.linalg.eigh(covarianceMatrix) #numpy, linear algebra, calculate eigenvalue of Hermitan (symmetric matrix)      

sorted_indices = np.argsort(eigenvalues)[::-1] #sort eigenvalues by descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices] #index of descending eigenvalues determines the sorting of eigenvectors

def display_eigenfaces():
    for i in range(0,50):
        eigenface = eigenvectors[:, i].reshape(128, 128)
        eigenface_normalized = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        
        skimage.io.imsave(f"./data/eigenfaces/{i+1}.jpg", (eigenface_normalized*255).astype(np.uint8))
        
#display_eigenfaces()

mean_face = skimage.io.imread("./data/meanFace.jpg")

def reconstruct_faces(eigenfaces,directory):

    top_eigenfaces = eigenvectors[:, :eigenfaces]
    
    for i in range(0,200):
        original_face = skimage.io.imread("./data/testing_hsv/"+f"{i+801}"+".jpg")
  
        projection = original_face.flatten() @ top_eigenfaces #project current test face onto top eigenfaces
        
        # Reconstruct using eigenfaces
        reconstructed_face = projection @ top_eigenfaces.T + mean_face.flatten()
        
        # Normalize and save reconstructed face
        reconstructed_normalized = (reconstructed_face.reshape(128, 128) - reconstructed_face.min()) / (reconstructed_face.max() - reconstructed_face.min())
        reconstructed_scaled = (reconstructed_normalized * 255).astype(np.uint8)
        skimage.io.imsave(f"./data/reconstructed_faces/{directory}/face_{i+801}_reconstructed.jpg", reconstructed_scaled)

# reconstruct_faces(eigenfaces=1,directory="1")

# for i in range(5,51,5):
#     reconstruct_faces(eigenfaces=i,directory=str(i))

def calculate_reconstruction_error():
    """
    Calculate and plot the total reconstruction error per pixel for different numbers of eigenfaces
    """
    Kvals = [1] + list(range(5, 51, 5))
    errors = []
    for k in Kvals:
        top_eigenfaces = eigenvectors[:, :k]
        total_error = 0
        for i in range(200):
            original_face = skimage.io.imread(f"./data/testing_hsv/{i+801}.jpg")
            original_flat = original_face.flatten().astype(np.float64)
            projection = original_flat @ top_eigenfaces
            reconstructed_face = projection @ top_eigenfaces.T + mean_face.flatten()
            total_error += np.sum((original_flat - reconstructed_face) ** 2)
        errors.append(total_error / (128*128*200))
    plt.figure(figsize=(10, 6))
    plt.plot(Kvals, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Eigenfaces (K)', fontsize=12)
    plt.ylabel('Reconstruction Error per Pixel', fontsize=12)
    plt.title('Reconstruction Error vs Number of Eigenfaces', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(Kvals)
    plt.tight_layout()
    plt.savefig('./data/reconstruction_error_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nReconstruction Error Results:")
    print("Eigenfaces\tError per Pixel")
    print("-" * 30)
    for k, error in zip(Kvals, errors):
        print(f"{k:8d}\t{error:.6f}")
    return Kvals, errors

# Calculate and plot reconstruction error
#eigenface_counts, errors = calculate_reconstruction_error()

def load_landmarks(landmark_dir, file_indices):
    """
    Load landmark data from .mat files
    Returns: landmarks matrix where each row is a flattened landmark vector
    """
    landmarks = []
    valid_indices = []
    
    for idx in file_indices:
        filepath = os.path.join(landmark_dir, f"{idx}.mat")
        if os.path.exists(filepath):
            try:
                mat_data = loadmat(filepath)
                # Extract landmark coordinates (usually stored as 'lm' or similar key)
                if 'lm' in mat_data:
                    lm = mat_data['lm']
                elif 'landmarks' in mat_data:
                    lm = mat_data['landmarks']
                else:
                    # Try to find the first non-metadata key
                    keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    if keys:
                        lm = mat_data[keys[0]]
                    else:
                        continue
                
                # Flatten the landmarks (x,y coordinates for each point)
                landmarks.append(lm.flatten())
                valid_indices.append(idx)
            except:
                print(f"Error loading {filepath}")
                continue
    
    return np.array(landmarks), valid_indices

def compute_landmark_eigen_warping():
    """
    Compute mean landmarks and eigen-warpings for training data
    """
    print("Loading training landmarks...")
    
    # Load training landmarks (indices 1-800)
    training_indices = list(range(1, 801))
    training_landmarks, valid_training_indices = load_landmarks("./data/training_landmarks", training_indices)
    
    print(f"Loaded {len(training_landmarks)} training landmark files")
    print(f"Landmark dimension: {training_landmarks.shape[1] if len(training_landmarks) > 0 else 'N/A'}")
    
    if len(training_landmarks) == 0:
        print("No landmark data loaded!")
        return None, None, None
    
    # Compute mean landmarks
    mean_landmarks = np.mean(training_landmarks, axis=0)
    print(f"Mean landmarks shape: {mean_landmarks.shape}")
    
    # Center the landmarks by subtracting mean
    centered_landmarks = training_landmarks - mean_landmarks
    
    # Compute covariance matrix for landmarks
    landmark_cov = centered_landmarks.T @ centered_landmarks
    
    # Compute eigenvalues and eigenvectors
    landmark_eigenvalues, landmark_eigenvectors = np.linalg.eigh(landmark_cov)
    
    # Sort by descending eigenvalues
    sorted_indices = np.argsort(landmark_eigenvalues)[::-1]
    landmark_eigenvalues = landmark_eigenvalues[sorted_indices]
    landmark_eigenvectors = landmark_eigenvectors[:, sorted_indices]
    
    print(f"Computed {len(landmark_eigenvalues)} eigen-warpings")
    
    return mean_landmarks, landmark_eigenvectors, landmark_eigenvalues

def display_eigen_warpings(mean_landmarks, eigenvectors, num_display=10):
    """
    Display the first 10 eigen-warpings by adding them to the mean
    """
    if mean_landmarks is None or eigenvectors is None:
        print("No landmark data available for display")
        return
    
    # Determine the number of landmark points
    landmark_dim = len(mean_landmarks)
    num_points = landmark_dim // 2  # Assuming x,y coordinates
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(min(num_display, eigenvectors.shape[1])):
        # Add eigen-warping to mean landmarks
        warped_landmarks = mean_landmarks + 3 * np.std(eigenvectors[:, i]) * eigenvectors[:, i]  # Scale for visibility
        
        # Reshape to (N, 2) for plotting
        mean_points = mean_landmarks.reshape(-1, 2)
        warped_points = warped_landmarks.reshape(-1, 2)
        
        # Plot
        ax = axes[i]
        ax.scatter(mean_points[:, 0], mean_points[:, 1], c='blue', s=20, alpha=0.6, label='Mean')
        ax.scatter(warped_points[:, 0], warped_points[:, 1], c='red', s=20, alpha=0.6, label='Warped')
        
        # Draw displacement vectors
        for j in range(len(mean_points)):
            ax.arrow(mean_points[j, 0], mean_points[j, 1], 
                    warped_points[j, 0] - mean_points[j, 0], 
                    warped_points[j, 1] - mean_points[j, 1],
                    head_width=2, head_length=2, fc='green', ec='green', alpha=0.5)
        
        ax.set_title(f'Eigen-warping {i+1}')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Invert y-axis to match image coordinate system (origin at top-left)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('./data/eigen_warpings.png', dpi=300, bbox_inches='tight')
    plt.show()

def reconstruct_test_landmarks_and_plot_error(mean_landmarks, eigenvectors):
    if mean_landmarks is None or eigenvectors is None: print("No landmark data available for reconstruction"); return
    print("Loading testing landmarks...")
    testing_indices = list(range(801, 1001))
    testing_landmarks, valid_testing_indices = load_landmarks("./data/testing_landmarks", testing_indices)
    print(f"Loaded {len(testing_landmarks)} testing landmark files")
    if len(testing_landmarks) == 0: print("No testing landmark data loaded!"); return
    Kvals = [1] + list(range(5, 51, 5))
    errors = []
    for K in Kvals:
        K = min(K, eigenvectors.shape[1])
        top_eigenvectors = eigenvectors[:, :K]
        total_error = 0
        for test_landmarks in testing_landmarks:
            centered_test = test_landmarks - mean_landmarks
            projection = centered_test @ top_eigenvectors
            reconstructed = projection @ top_eigenvectors.T + mean_landmarks
            original_points = test_landmarks.reshape(-1, 2)
            reconstructed_points = reconstructed.reshape(-1, 2)
            distances = np.sqrt(np.sum((original_points - reconstructed_points)**2, axis=1))
            total_error += np.mean(distances)
        errors.append(total_error / len(testing_landmarks))
    plt.figure(figsize=(10, 6))
    plt.plot(Kvals, errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Eigen-warpings (K)', fontsize=12)
    plt.ylabel('Average Reconstruction Error (pixels)', fontsize=12)
    plt.title('Landmark Reconstruction Error vs Number of Eigen-warpings', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(Kvals)
    plt.tight_layout()
    plt.savefig('./data/landmark_reconstruction_error_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nLandmark Reconstruction Error Results:")
    print("Eigen-warpings\tAvg Error (pixels)")
    print("-" * 35)
    for k, error in zip(Kvals, errors):
        print(f"{k:11d}\t{error:.6f}")
    return Kvals, errors

def run_eigen_warping_analysis():
    """
    Main function to run the complete eigen-warping analysis
    """
    print("=== Eigen-warping Analysis for Landmarks ===")
    
    # Step 1: Compute eigen-warpings
    mean_landmarks, eigenvectors, eigenvalues = compute_landmark_eigen_warping()
    
    if mean_landmarks is None:
        print("Failed to load landmark data. Please check the file paths and formats.")
        return
    
    # Step 2: Display first 10 eigen-warpings
    print("\nDisplaying first 10 eigen-warpings...")
    display_eigen_warpings(mean_landmarks, eigenvectors, num_display=10)
    
    # Step 3: Reconstruct test landmarks and plot error
    print("\nReconstructing test landmarks and computing errors...")
    K_values, errors = reconstruct_test_landmarks_and_plot_error(mean_landmarks, eigenvectors)
    
    print("\n=== Analysis Complete ===")
    return mean_landmarks, eigenvectors, eigenvalues, K_values, errors

# Run the eigen-warping analysis
#landmark_results = run_eigen_warping_analysis()

# Import the warping function
from mywarper import warp

def combined_reconstruction_analysis():
    print("=== Combined Reconstruction Analysis (Efficient) ===")
    mean_landmarks, landmark_eigenvectors, landmark_eigenvalues = compute_landmark_eigen_warping()
    if mean_landmarks is None: print("Failed to load landmark data."); return
    training_indices = list(range(1, 801))
    training_landmarks, valid_training_indices = load_landmarks("./data/training_landmarks", training_indices)
    if len(training_landmarks) == 0: print("No training landmarks loaded!"); return
    aligned_training_images = []
    mean_landmarks_reshaped = mean_landmarks.reshape(-1, 2)
    for i, idx in enumerate(valid_training_indices):
        if i >= 799: break
        try:
            img_path = f"./data/training_hsv/{idx}.jpg"
            if os.path.exists(img_path):
                img = skimage.io.imread(img_path)
                if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
                landmarks_reshaped = training_landmarks[i].reshape(-1, 2)
                aligned_img = warp(img, landmarks_reshaped, mean_landmarks_reshaped)
                aligned_training_images.append(aligned_img.flatten())
        except: continue
    aligned_training_images = np.array(aligned_training_images)
    aligned_mean_face = np.mean(aligned_training_images, axis=0)
    centered_aligned = aligned_training_images - aligned_mean_face
    aligned_cov = centered_aligned.T @ centered_aligned
    aligned_eigenvalues, aligned_eigenvectors = np.linalg.eigh(aligned_cov)
    sorted_indices = np.argsort(aligned_eigenvalues)[::-1]
    aligned_eigenvalues = aligned_eigenvalues[sorted_indices]
    aligned_eigenvectors = aligned_eigenvectors[:, sorted_indices]
    testing_indices = list(range(801, 1001))
    testing_landmarks, valid_testing_indices = load_landmarks("./data/testing_landmarks", testing_indices)
    if len(testing_landmarks) == 0: print("No testing landmarks loaded!"); return
    nL, Kvals = 10, [1]+list(range(5,51,5))
    nL = min(nL, landmark_eigenvectors.shape[1])
    Lvecs = landmark_eigenvectors[:, :nL]
    mean_landmarks_reshaped = mean_landmarks.reshape(-1,2)
    errors = []; saved_examples = []
    for K in Kvals:
        K = min(K, aligned_eigenvectors.shape[1])
        Avecs = aligned_eigenvectors[:, :K]
        total_error = 0; num_processed = 0
        for i, test_idx in enumerate(valid_testing_indices):
            if i >= 200: break
            try:
                test_img_path = f"./data/testing_hsv/{test_idx}.jpg"
                if not os.path.exists(test_img_path): continue
                original_img = skimage.io.imread(test_img_path)
                if len(original_img.shape) == 2: original_img_3d = np.expand_dims(original_img,2)
                else: original_img_3d = original_img
                test_landmarks = testing_landmarks[i]
                centered_test_landmarks = test_landmarks - mean_landmarks
                landmark_proj = centered_test_landmarks @ Lvecs
                recon_landmarks = landmark_proj @ Lvecs.T + mean_landmarks
                test_landmarks_reshaped = test_landmarks.reshape(-1,2)
                warped_to_mean = warp(original_img_3d, test_landmarks_reshaped, mean_landmarks_reshaped)
                warped_flat = warped_to_mean.flatten()
                centered_warped = warped_flat - aligned_mean_face
                appearance_proj = centered_warped @ Avecs
                recon_appearance = appearance_proj @ Avecs.T + aligned_mean_face
                recon_img_2d = recon_appearance.reshape(128,128)
                recon_img_3d = np.expand_dims(recon_img_2d,2)
                recon_landmarks_reshaped = recon_landmarks.reshape(-1,2)
                final_recon = warp(recon_img_3d, mean_landmarks_reshaped, recon_landmarks_reshaped)
                if len(original_img.shape) == 2:
                    original_flat = original_img.flatten().astype(np.float64)
                    recon_flat = final_recon[:,:,0].flatten().astype(np.float64)
                else:
                    original_flat = original_img.flatten().astype(np.float64)
                    recon_flat = final_recon.flatten().astype(np.float64)
                total_error += np.sum((original_flat-recon_flat)**2)
                num_processed += 1
                if K == Kvals[0] and len(saved_examples) < 20:
                    saved_examples.append({'original': original_img, 'reconstructed': final_recon[:,:,0] if final_recon.shape[2]==1 else final_recon, 'test_idx': test_idx})
            except: continue
        if num_processed > 0:
            pixels_per_image = 128*128
            errors.append(total_error/(pixels_per_image*num_processed))
        else:
            errors.append(float('inf'))
    plt.figure(figsize=(12,6))
    plt.plot(Kvals, errors, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Eigenfaces (K)', fontsize=12)
    plt.ylabel('Reconstruction Error per Pixel', fontsize=12)
    plt.title('Combined Reconstruction Error vs Number of Eigenfaces\n(10 Eigen-warpings + K Eigenfaces)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(Kvals)
    plt.tight_layout()
    plt.savefig('./data/combined_reconstruction_error_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    if saved_examples:
        n_examples = min(20, len(saved_examples))
        fig, axes = plt.subplots(4, 10, figsize=(20, 8))
        for i in range(min(10, n_examples)):
            axes[0, i].imshow(saved_examples[i]['original'], cmap='gray')
            axes[0, i].set_title(f'Original {saved_examples[i]["test_idx"]}', fontsize=8)
            axes[0, i].axis('off')
            axes[1, i].imshow(saved_examples[i]['reconstructed'], cmap='gray')
            axes[1, i].set_title(f'Reconstructed {saved_examples[i]["test_idx"]}', fontsize=8)
            axes[1, i].axis('off')
        if n_examples > 10:
            for i in range(10, n_examples):
                col = i - 10
                axes[2, col].imshow(saved_examples[i]['original'], cmap='gray')
                axes[2, col].set_title(f'Original {saved_examples[i]["test_idx"]}', fontsize=8)
                axes[2, col].axis('off')
                axes[3, col].imshow(saved_examples[i]['reconstructed'], cmap='gray')
                axes[3, col].set_title(f'Reconstructed {saved_examples[i]["test_idx"]}', fontsize=8)
                axes[3, col].axis('off')
        for row in range(4):
            for col in range(10):
                if (row < 2 and col >= min(10, n_examples)) or (row >= 2 and col >= max(0, n_examples - 10)):
                    axes[row, col].axis('off')
        plt.suptitle('Original vs Reconstructed Faces\n(Rows 1&3: Originals, Rows 2&4: Reconstructed)', fontsize=14)
        plt.tight_layout()
        plt.savefig('./data/combined_reconstruction_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
    print("\n=== Combined Reconstruction Results ===")
    print("Eigenfaces\tError per Pixel")
    print("-" * 30)
    for k, error in zip(Kvals, errors):
        if error != float('inf'): print(f"{k:8d}\t{error:.6f}")
        else: print(f"{k:8d}\tError")
    return Kvals, errors, saved_examples

# Run the combined reconstruction analysis
print("\nStarting combined reconstruction analysis...")
combined_results = combined_reconstruction_analysis()

def synthesize_random_faces(num_faces=50):
    print("=== Random Face Synthesis (Efficient) ===")
    mean_landmarks, landmark_eigenvectors, landmark_eigenvalues = compute_landmark_eigen_warping()
    if mean_landmarks is None:
        print("Failed to load landmark eigenspace"); return
    training_indices = list(range(1, 801))
    training_landmarks, valid_training_indices = load_landmarks("./data/training_landmarks", training_indices)
    if len(training_landmarks) == 0:
        print("No training landmarks loaded!"); return
    aligned_training_images = []
    mean_landmarks_reshaped = mean_landmarks.reshape(-1, 2)
    for i, idx in enumerate(valid_training_indices):
        if i >= 799: break
        try:
            img_path = f"./data/training_hsv/{idx}.jpg"
            if os.path.exists(img_path):
                img = skimage.io.imread(img_path)
                if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
                landmarks_reshaped = training_landmarks[i].reshape(-1, 2)
                aligned_img = warp(img, landmarks_reshaped, mean_landmarks_reshaped)
                aligned_training_images.append(aligned_img.flatten())
        except: continue
    aligned_training_images = np.array(aligned_training_images)
    aligned_mean_face = np.mean(aligned_training_images, axis=0)
    centered_aligned = aligned_training_images - aligned_mean_face
    aligned_cov = centered_aligned.T @ centered_aligned
    aligned_eigenvalues, aligned_eigenvectors = np.linalg.eigh(aligned_cov)
    sorted_indices = np.argsort(aligned_eigenvalues)[::-1]
    aligned_eigenvalues = aligned_eigenvalues[sorted_indices]
    aligned_eigenvectors = aligned_eigenvectors[:, sorted_indices]
    nL, nA = 10, 50
    nL = min(nL, landmark_eigenvectors.shape[1]); nA = min(nA, aligned_eigenvectors.shape[1])
    Lvecs = landmark_eigenvectors[:, :nL]; Lstd = np.sqrt(np.maximum(landmark_eigenvalues[:nL], 1e-10))
    Avecs = aligned_eigenvectors[:, :nA]; Astd = np.sqrt(np.maximum(aligned_eigenvalues[:nA], 1e-10))
    Lcoeffs = np.random.randn(num_faces, nL)
    Acoeffs = np.random.randn(num_faces, nA)
    # Synthesize landmarks: mean + (coeffs * std) @ eigvecs.T
    synth_landmarks = mean_landmarks + (Lcoeffs * Lstd).dot(Lvecs.T)
    # Synthesize appearance: mean + (coeffs * std) @ eigvecs.T
    synth_appearance = aligned_mean_face + (Acoeffs * Astd).dot(Avecs.T)
    synth_imgs = []
    for i in range(num_faces):
        try:
            img2d = synth_appearance[i].reshape(128,128); img3d = np.expand_dims(img2d,2)
            lm = synth_landmarks[i].reshape(-1,2)
            face = warp(img3d, mean_landmarks_reshaped, lm)
            synth_imgs.append(face[:,:,0])
        except: synth_imgs.append(np.random.randint(0,255,(128,128)).astype(np.uint8))
    synth_imgs = np.array(synth_imgs)
    # Display
    fig, axes = plt.subplots(5,10,figsize=(20,10))
    for i in range(num_faces):
        r,c = divmod(i,10)
        f = synth_imgs[i]; f = (f-f.min())/(f.max()-f.min()+1e-8)
        axes[r,c].imshow(f, cmap='gray'); axes[r,c].set_title(f'{i+1}',fontsize=8); axes[r,c].axis('off')
    plt.suptitle('50 Synthesized Random Faces (Efficient)',fontsize=16)
    plt.tight_layout(); plt.savefig('./data/synthesized_faces.png',dpi=300,bbox_inches='tight'); plt.show()
    # Save
    os.makedirs('./data/synthesized_faces',exist_ok=True)
    for i,face in enumerate(synth_imgs):
        f = (face-face.min())/(face.max()-face.min()+1e-8); f = (f*255).astype(np.uint8)
        skimage.io.imsave(f'./data/synthesized_faces/synthesized_face_{i+1:03d}.jpg',f)
    print(f'Saved {num_faces} synthesized faces to ./data/synthesized_faces')
    return synth_imgs, Lcoeffs, Acoeffs

#synthesis_results = synthesize_random_faces(num_faces=50)

def load_gender_data():
    def load_imgs_lms(img_dir, lm_dir, indices):
        imgs, lms = [], []
        for idx in indices:
            # Use 6-digit zero-padded format to match actual filenames
            img_path = os.path.join(img_dir, f"{idx:06d}.jpg")
            lm_path = os.path.join(lm_dir, f"{idx:06d}.mat")
            if os.path.exists(img_path) and os.path.exists(lm_path):
                try:
                    img = skimage.io.imread(img_path)
                    if len(img.shape) == 2: img = np.expand_dims(img, 2)
                    imgs.append(img)
                    mat = loadmat(lm_path)
                    if 'lm' in mat: lms.append(mat['lm'].flatten())
                    elif 'landmarks' in mat: lms.append(mat['landmarks'].flatten())
                    else:
                        keys = [k for k in mat.keys() if not k.startswith('__')]
                        if keys: lms.append(mat[keys[0]].flatten())
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        return imgs, np.array(lms)
    
    # Get file indices from actual filenames (remove .jpg extension and convert to int)
    male_files = [f for f in os.listdir('./data/male_images') if f.endswith('.jpg')]
    female_files = [f for f in os.listdir('./data/female_images') if f.endswith('.jpg')]
    
    male_idx = [int(f.split('.')[0]) for f in male_files]
    female_idx = [int(f.split('.')[0]) for f in female_files]
    
    print(f"Found {len(male_idx)} male images and {len(female_idx)} female images")
    
    # Combine all indices and shuffle
    all_male_idx = male_idx.copy()
    all_female_idx = female_idx.copy()
    random.shuffle(all_male_idx)
    random.shuffle(all_female_idx)
    
    # Use 800 total faces for training, 200 for testing as per instructions
    # Calculate proportional split to maintain gender balance
    total_faces = len(all_male_idx) + len(all_female_idx)
    train_size = 800
    test_size = 200
    
    # Proportional split
    male_train_size = int(train_size * len(all_male_idx) / total_faces)
    female_train_size = train_size - male_train_size
    
    m_train = all_male_idx[:male_train_size]
    m_test = all_male_idx[male_train_size:]
    f_train = all_female_idx[:female_train_size]
    f_test = all_female_idx[female_train_size:]
    
    print(f"Training: {len(m_train)} male, {len(f_train)} female (total: {len(m_train)+len(f_train)})")
    print(f"Testing: {len(m_test)} male, {len(f_test)} female (total: {len(m_test)+len(f_test)})")
    
    X_male_train, L_male_train = load_imgs_lms('./data/male_images', './data/male_landmarks', m_train)
    X_male_test, L_male_test = load_imgs_lms('./data/male_images', './data/male_landmarks', m_test)
    X_fem_train, L_fem_train = load_imgs_lms('./data/female_images', './data/female_landmarks', f_train)
    X_fem_test, L_fem_test = load_imgs_lms('./data/female_images', './data/female_landmarks', f_test)
    
    print(f"Loaded: {len(X_male_train)} male train, {len(X_male_test)} male test")
    print(f"Loaded: {len(X_fem_train)} female train, {len(X_fem_test)} female test")
    
    X_train = X_male_train + X_fem_train
    X_test = X_male_test + X_fem_test
    L_train = np.vstack([L_male_train, L_fem_train])
    L_test = np.vstack([L_male_test, L_fem_test])
    y_train = np.array([0]*len(X_male_train) + [1]*len(X_fem_train))
    y_test = np.array([0]*len(X_male_test) + [1]*len(X_fem_test))
    
    print(f"Final dataset: {len(X_train)} train, {len(X_test)} test")
    return X_train, L_train, y_train, X_test, L_test, y_test

def extract_features(X, L, mean_landmarks, landmark_eigvecs, mean_face, face_eigvecs):
    geom = (L - mean_landmarks) @ landmark_eigvecs[:, :10]
    app = []
    for img, lm in zip(X, L):
        try:
            img_aligned = warp(img, lm.reshape(-1,2), mean_landmarks.reshape(-1,2))
            flat = img_aligned.flatten() - mean_face
            app.append(flat @ face_eigvecs[:, :50])
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Use zero vector as fallback
            app.append(np.zeros(50))
    app = np.array(app)
    return geom, app

def fisher_analysis():
    X_train, L_train, y_train, X_test, L_test, y_test = load_gender_data()
    
    # Check if we have data
    if len(X_train) == 0 or len(X_test) == 0:
        print("No data loaded! Check if male_images/female_images directories exist and contain files.")
        return
    
    print(f"Starting Fisher analysis with {len(X_train)} training and {len(X_test)} test samples")
    
    mean_landmarks = np.mean(L_train, axis=0)
    centered_landmarks = L_train - mean_landmarks
    cov_lm = centered_landmarks.T @ centered_landmarks
    eigval_lm, eigvec_lm = np.linalg.eigh(cov_lm)
    idx = np.argsort(eigval_lm)[::-1]
    eigvec_lm = eigvec_lm[:, idx]
    
    # Appearance PCA
    print("Computing appearance PCA...")
    aligned_imgs = []
    for img, lm in zip(X_train, L_train):
        try:
            aligned = warp(img, lm.reshape(-1,2), mean_landmarks.reshape(-1,2))
            aligned_imgs.append(aligned.flatten())
        except Exception as e:
            print(f"Error warping image: {e}")
            continue
    
    if len(aligned_imgs) == 0:
        print("No images could be aligned! Check warping function.")
        return
    
    aligned_imgs = np.array(aligned_imgs)
    mean_face = np.mean(aligned_imgs, axis=0)
    centered_imgs = aligned_imgs - mean_face
    cov_face = centered_imgs.T @ centered_imgs
    eigval_face, eigvec_face = np.linalg.eigh(cov_face)
    idxf = np.argsort(eigval_face)[::-1]
    eigvec_face = eigvec_face[:, idxf]
    
    # Feature extraction
    print("Extracting features...")
    geom_train, app_train = extract_features(X_train, L_train, mean_landmarks, eigvec_lm, mean_face, eigvec_face)
    geom_test, app_test = extract_features(X_test, L_test, mean_landmarks, eigvec_lm, mean_face, eigvec_face)
    
    if len(geom_train) == 0 or len(app_train) == 0:
        print("No features extracted! Check feature extraction function.")
        return
    
    # Data validation and preprocessing
    print("Validating and preprocessing data...")
    
    # Check for NaN and infinite values
    if np.any(np.isnan(geom_train)) or np.any(np.isinf(geom_train)):
        print("Warning: NaN or infinite values in geometry features. Replacing with zeros.")
        geom_train = np.nan_to_num(geom_train, nan=0.0, posinf=0.0, neginf=0.0)
        geom_test = np.nan_to_num(geom_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(app_train)) or np.any(np.isinf(app_train)):
        print("Warning: NaN or infinite values in appearance features. Replacing with zeros.")
        app_train = np.nan_to_num(app_train, nan=0.0, posinf=0.0, neginf=0.0)
        app_test = np.nan_to_num(app_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Add small regularization to avoid singular matrices
    geom_train = geom_train + 1e-10 * np.random.randn(*geom_train.shape)
    app_train = app_train + 1e-10 * np.random.randn(*app_train.shape)
    
    # Ensure data types are correct
    geom_train = geom_train.astype(np.float64)
    app_train = app_train.astype(np.float64)
    geom_test = geom_test.astype(np.float64)
    app_test = app_test.astype(np.float64)
    
    Xtr = np.hstack([geom_train, app_train]); Xte = np.hstack([geom_test, app_test])
    
    # Fisher face (combined) - 1D LDA + 1D PCA for 2D visualization
    try:
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(Xtr, y_train)
        fisher_combined = lda.transform(np.vstack([Xtr, Xte]))
        
        # Add PCA component for 2D visualization
        pca_combined = PCA(n_components=1)
        pca_combined.fit(Xtr)
        pca_combined_comp = pca_combined.transform(np.vstack([Xtr, Xte]))
        
        # Combine Fisher and PCA for 2D visualization
        proj_combined_2d = np.column_stack([fisher_combined, pca_combined_comp])
        
        y_pred = lda.predict(Xte)
        err = np.mean(y_pred != y_test)
        print(f'Fisher face (combined) test error: {err:.3f}')
    except Exception as e:
        print(f"Error in combined Fisher analysis: {e}")
        return
    
    # Fisher for geometry - 1D LDA + 1D PCA for 2D visualization
    try:
        lda_g = LinearDiscriminantAnalysis(n_components=1)
        lda_g.fit(geom_train, y_train)
        fisher_g = lda_g.transform(np.vstack([geom_train, geom_test]))
        
        # Add PCA component for 2D visualization
        pca_g = PCA(n_components=1)
        pca_g.fit(geom_train)
        pca_g_comp = pca_g.transform(np.vstack([geom_train, geom_test]))
        
        # Combine Fisher and PCA for 2D visualization
        proj_g_2d = np.column_stack([fisher_g, pca_g_comp])
        
    except Exception as e:
        print(f"Error in geometry Fisher analysis: {e}")
        # Fallback to PCA if LDA fails
        print("Falling back to PCA for geometry visualization...")
        pca_g = PCA(n_components=2)
        pca_g.fit(geom_train)
        proj_g_2d = pca_g.transform(np.vstack([geom_train, geom_test]))
    
    # Fisher for appearance - 1D LDA + 1D PCA for 2D visualization
    try:
        lda_a = LinearDiscriminantAnalysis(n_components=1)
        lda_a.fit(app_train, y_train)
        fisher_a = lda_a.transform(np.vstack([app_train, app_test]))
        
        # Add PCA component for 2D visualization
        pca_a = PCA(n_components=1)
        pca_a.fit(app_train)
        pca_a_comp = pca_a.transform(np.vstack([app_train, app_test]))
        
        # Combine Fisher and PCA for 2D visualization
        proj_a_2d = np.column_stack([fisher_a, pca_a_comp])
        
    except Exception as e:
        print(f"Error in appearance Fisher analysis: {e}")
        # Fallback to PCA if LDA fails
        print("Falling back to PCA for appearance visualization...")
        pca_a = PCA(n_components=2)
        pca_a.fit(app_train)
        proj_a_2d = pca_a.transform(np.vstack([app_train, app_test]))
    
    # Visualization
    plt.figure(figsize=(15,12))
    
    # Combined Fisher Face - Male Discriminating Direction
    plt.subplot(2,2,1)
    plt.scatter(proj_combined_2d[:len(y_train),0], proj_combined_2d[:len(y_train),1], c=y_train, cmap='coolwarm', label='Train', alpha=0.6)
    plt.scatter(proj_combined_2d[len(y_train):,0], proj_combined_2d[len(y_train):,1], c=y_test, cmap='coolwarm', marker='x', label='Test', alpha=0.6)
    plt.title(f'Male-Discriminating Fisher Face 2D\n(Geometry + Appearance)\nError: {err:.3f}')
    plt.xlabel('Fisher Component')
    plt.ylabel('PCA Component')
    # Add custom legend with gender labels
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='Female'),
                      Patch(facecolor='blue', alpha=0.6, label='Male')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Geometry Fisher Face - Male Discriminating Direction
    plt.subplot(2,2,2)
    plt.scatter(proj_g_2d[:len(y_train),0], proj_g_2d[:len(y_train),1], c=y_train, cmap='coolwarm', label='Train', alpha=0.6)
    plt.scatter(proj_g_2d[len(y_train):,0], proj_g_2d[len(y_train):,1], c=y_test, cmap='coolwarm', marker='x', label='Test', alpha=0.6)
    plt.title('Male-Discriminating Fisher Face Geometry 2D\n(Key-points/Landmarks)')
    plt.xlabel('Fisher Component')
    plt.ylabel('PCA Component')
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Appearance Fisher Face - Male Discriminating Direction
    plt.subplot(2,2,3)
    plt.scatter(proj_a_2d[:len(y_train),0], proj_a_2d[:len(y_train),1], c=y_train, cmap='coolwarm', label='Train', alpha=0.6)
    plt.scatter(proj_a_2d[len(y_train):,0], proj_a_2d[len(y_train):,1], c=y_test, cmap='coolwarm', marker='x', label='Test', alpha=0.6)
    plt.title('Male-Discriminating Fisher Face Appearance 2D\n(Aligned to Mean Position)')
    plt.xlabel('Fisher Component')
    plt.ylabel('PCA Component')
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Summary statistics
    plt.subplot(2,2,4)
    plt.text(0.1, 0.8, f'Male-Discriminating Fisher Face Analysis', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f'Training samples: {len(y_train)}', fontsize=12)
    plt.text(0.1, 0.6, f'Test samples: {len(y_test)}', fontsize=12)
    plt.text(0.1, 0.5, f'Geometry features: 10D', fontsize=12)
    plt.text(0.1, 0.4, f'Appearance features: 50D', fontsize=12)
    plt.text(0.1, 0.3, f'Combined features: 60D', fontsize=12)
    plt.text(0.1, 0.2, f'Combined error rate: {err:.3f}', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.1, f'All projections: Fisher + PCA', fontsize=12)
    plt.text(0.1, 0.0, f'Red = Female, Blue = Male', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./data/male_discriminating_fisher_scatter.png', dpi=200)
    plt.show()
    
    # Save Fisher faces as images
    print("\nSaving male-discriminating Fisher face as image...")
    try:
        # Get the Fisher discriminant directions from the trained LDA models
        fisher_direction_geom = lda_g.coef_[0]  # Geometry Fisher direction
        fisher_direction_app = lda_a.coef_[0]   # Appearance Fisher direction
        fisher_direction_combined = lda.coef_[0]  # Combined Fisher direction
        
        print(f"Fisher direction shapes:")
        print(f"  Geometry: {fisher_direction_geom.shape}")
        print(f"  Appearance: {fisher_direction_app.shape}")
        print(f"  Combined: {fisher_direction_combined.shape}")
        
        # Normalize the directions
        fisher_direction_geom = fisher_direction_geom / np.linalg.norm(fisher_direction_geom)
        fisher_direction_app = fisher_direction_app / np.linalg.norm(fisher_direction_app)
        fisher_direction_combined = fisher_direction_combined / np.linalg.norm(fisher_direction_combined)
        
        # Reconstruct only the male-discriminating Fisher face
        # For geometry: reconstruct landmarks using the Fisher direction
        male_fisher_landmarks = mean_landmarks + fisher_direction_geom @ eigvec_lm[:, :10].T
        
        # For appearance: reconstruct aligned face using the Fisher direction
        male_fisher_appearance = mean_face + fisher_direction_app @ eigvec_face[:, :50].T
        
        print(f"Reconstructed shapes:")
        print(f"  Male appearance: {male_fisher_appearance.shape}")
        
        # For combined: reconstruct using both geometry and appearance
        # Split the combined direction into geometry and appearance parts
        geom_part = fisher_direction_combined[:10]  # First 10 components are geometry
        app_part = fisher_direction_combined[10:]   # Last 50 components are appearance
        
        male_combined_landmarks = mean_landmarks + geom_part @ eigvec_lm[:, :10].T
        male_combined_appearance = mean_face + app_part @ eigvec_face[:, :50].T
        
        print(f"Combined appearance shapes:")
        print(f"  Male combined: {male_combined_appearance.shape}")
        
        # Save geometry Fisher face (landmark plot) - MALE ONLY
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Male geometry Fisher face
        male_lm_2d = male_fisher_landmarks.reshape(-1, 2)
        axes[0].scatter(male_lm_2d[:, 0], male_lm_2d[:, 1], c='blue', s=20)
        axes[0].set_title('Male-Discriminating Fisher Face (Geometry)')
        axes[0].set_aspect('equal')
        axes[0].invert_yaxis()
        
        # Male combined Fisher face
        male_comb_lm_2d = male_combined_landmarks.reshape(-1, 2)
        axes[1].scatter(male_comb_lm_2d[:, 0], male_comb_lm_2d[:, 1], c='blue', s=20)
        axes[1].set_title('Male-Discriminating Fisher Face (Combined)')
        axes[1].set_aspect('equal')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('./data/male_fisher_faces_geometry.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save appearance Fisher face as actual 128x128 image - MALE ONLY
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Male appearance Fisher face - reshape to 128x128x3 then convert to grayscale
        if male_fisher_appearance.size == 49152:  # 3-channel image
            male_face = male_fisher_appearance.reshape(128, 128, 3)
            # Convert to grayscale by taking the mean across channels
            male_face = np.mean(male_face, axis=2)
        else:
            male_face = male_fisher_appearance.reshape(128, 128)
        male_face_norm = (male_face - male_face.min()) / (male_face.max() - male_face.min())
        axes[0].imshow(male_face_norm, cmap='gray')
        axes[0].set_title('Male-Discriminating Fisher Face (Appearance)')
        axes[0].axis('off')
        
        # Male combined Fisher face - reshape to 128x128x3 then convert to grayscale
        if male_combined_appearance.size == 49152:  # 3-channel image
            male_comb_face = male_combined_appearance.reshape(128, 128, 3)
            # Convert to grayscale by taking the mean across channels
            male_comb_face = np.mean(male_comb_face, axis=2)
        else:
            male_comb_face = male_combined_appearance.reshape(128, 128)
        male_comb_face_norm = (male_comb_face - male_comb_face.min()) / (male_comb_face.max() - male_comb_face.min())
        axes[1].imshow(male_comb_face_norm, cmap='gray')
        axes[1].set_title('Male-Discriminating Fisher Face (Combined)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('./data/male_fisher_faces_appearance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save individual male Fisher face images as 128x128 files
        skimage.io.imsave('./data/male_discriminating_fisher_face_appearance.jpg', 
                         (male_face_norm * 255).astype(np.uint8))
        skimage.io.imsave('./data/male_discriminating_fisher_face_combined.jpg', 
                         (male_comb_face_norm * 255).astype(np.uint8))
        
        print("Male-discriminating Fisher face saved as:")
        print("- ./data/male_fisher_faces_geometry.png (landmark plots)")
        print("- ./data/male_fisher_faces_appearance.png (128x128 face images)")
        print("- ./data/male_discriminating_fisher_face_appearance.jpg (128x128)")
        print("- ./data/male_discriminating_fisher_face_combined.jpg (128x128)")
        
    except Exception as e:
        print(f"Error saving Fisher face: {e}")
        import traceback
        traceback.print_exc()

#print('\nStarting Fisher face gender analysis...')
fisher_analysis()
