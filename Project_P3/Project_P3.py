import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Tuple

class EigenfaceRecognizer:
    def __init__(self, threshold_0: int = 10, threshold_1: int = 8): # Sensitive for specific threshold hyperparameters
        self.mean_face = None
        self.eigenfaces = None  # U matrix from SVD
        self.coordinates = None  # Projection coordinates for known faces
        self.image_size = None
        self.threshold_0 = threshold_0  # Threshold for known/unknown face
        self.threshold_1 = threshold_1  # Threshold for face/non-face

    def load_and_flatten_image(self, image_path):
        """Load an image and flatten it to a column vector."""
        # Conversion since the images are initially in 'RGB' mode. Namely, 320x320x3
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        self.image_size = img.size
        # img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        return np.array(img).flatten() # m x n -> (mn) x 1 = M x 1

    def train(self, faces_folder): # eps_0 = 10, eps_1 = 8
        """Train the eigenface recognizer with a set of known faces."""
        #self.threshold_0 = threshold_0  # Threshold for known/unknown face
        #self.threshold_1 = threshold_1  # Threshold for face/non-face
        
        # Load all face images
        face_files = list(Path(faces_folder).glob('*.jpg'))
        N = len(face_files) # number of samples (dim(Sample space))
        if N == 0: raise ValueError(f"No jpg files found in {faces_folder}")
         
        # Create matrix S where each column is a flattened face image
        first_image = self.load_and_flatten_image(face_files[0])
        M = len(first_image) # 102_400
        S = np.zeros((M, N)) # 102_400 x 36
        
        # for i, face_file in enumerate(face_files): 
        #     # construct matrix S by horizontal stacking the (grayscaled) images
        #     S[:, i] = self.load_and_flatten_image(face_file) # [{f_i}_{i \in [N]}]
            
        # Load and verify dimensions of each image
        print(f"Loading {N} images...")
        for i, face_file in enumerate(face_files):
            try:
                face_vector = self.load_and_flatten_image(face_file)
                if len(face_vector) != M:
                    raise ValueError(f"Image {face_file} has incorrect dimensions")
                S[:, i] = face_vector
                print(f"Successfully loaded image {i+1}/{N}: {face_file.name}")
            except Exception as e:
                raise ValueError(f"Error processing {face_file}: {str(e)}")
              
        # Compute mean face and subtract from each image
        self.mean_face = np.mean(S, axis=1) # Python follows row-major order
        A = S - self.mean_face.reshape(-1, 1) # elements: a_i = f_i − \bar{f}, for each i \in [N]
        
        # Perform reduced SVD on matrix A
        print("Computing SVD...")
        # shapes are dim(U) = (M, k) and dim(Vt) = (k, N), where k = min(M, N) (= N).
        U, sigma, Vt = np.linalg.svd(A, full_matrices=False) # sigma is an array containing N-r zeroes. 
                                                             # sigma != S obviously
        # Get rank (number of non-zero singular values)
        r = np.sum(sigma > 1e-10) # r = rank(A) <= min(M, N) (= N)
        print(f"Rank of face space: {r}") # rank(A) = np.linalg.matrix_rank(A) = 35
        self.eigenfaces = U[:, :r] # 102_400 x 35

        # Compute coordinates for each known face
        self.coordinates = self.eigenfaces.T @ A # x =  U[:, :r].T @ (f-\bar{f}), dim(x) = (r x N)
        self.known_faces = [f.stem for f in face_files]  # Store filenames without extension
        print("Training completed successfully")

        # Store the average reconstruction error for the training set
        # reconstructions = self.eigenfaces @ self.coordinates
        # self.avg_train_error = np.mean([np.linalg.norm(A[:, i] - reconstructions[:, i]) / np.sqrt(M) 
        #                               for i in range(N)])
        # print(f"Average training reconstruction error: {self.avg_train_error:.4f}")
        
    def classify_image(self, image_path) -> Tuple[str, float, float | None]:
        """Classify a new image as either not a face, unknown face, or a known face."""

        if self.eigenfaces is None:
            raise ValueError("Model not trained yet")
            
        # Load test image and error handle it
        try:
            f = self.load_and_flatten_image(image_path)
        except Exception as e:
            raise ValueError(f"Error loading test image {image_path}: {str(e)}")
            
        M = len(f)  # Image dimensions for normalization  

        # Normalize test image
        f_normalized = f - self.mean_face # matrix A for Testing Set
        
        # Project onto face subspace
        x = self.eigenfaces.T @ f_normalized # construct x with reduced U matrix from the training step
        f_p = self.eigenfaces @ x # f_p = U[:,:r]*x
        
        # Compute normalized distance from face space
        epsilon_f = np.linalg.norm(f_normalized - f_p) / np.sqrt(M) # eps_f = ||(f − \bar{f}) − f_p||_2
        
        # Compute normalized distances to all known faces
        
        # eps_i = ||x-x_i||_2 for each i \in [N]
        # distances = np.linalg.norm(self.coordinates - x.reshape(-1, 1), axis=0) # row-wise norm for R^N feature space
        distances = np.array([np.linalg.norm(self.coordinates[:, i] - x) / np.sqrt(M)\
                              for i in range(len(self.known_faces))])
        
        # obtain minimum eps_i for comparison with eps_0
        min_distance_idx = np.argmin(distances) # i_min = argmin_{i \in [N]}(eps_i) is the closest image
        min_distance = distances[min_distance_idx] # eps_min = min_{i \in [N]}(eps_i)
        

        print(f"VECTOR e_f:\t{epsilon_f}\n")
        print(f"VECTOR min(e_i):\t{min_distance}\n")

        if epsilon_f > self.threshold_1: 
            return "Not a face", epsilon_f, None

        if min_distance > self.threshold_0:
            return "Unknown face", epsilon_f, min_distance
        else:
            return self.known_faces[min_distance_idx], epsilon_f, min_distance

    def visualize_eigenface(self, index, output_path=None):
        """Visualize a specific eigenface."""
        if self.eigenfaces is None:
            raise ValueError("Model not trained yet")
            
        eigenface = self.eigenfaces[:, index].reshape(self.image_size)
        
        # Normalize to 0-255 range
        eigenface = ((eigenface - eigenface.min()) * 255 / 
                    (eigenface.max() - eigenface.min())).astype(np.uint8)
        
        img = Image.fromarray(eigenface)
        if output_path:
            img.save(output_path)
        return img

def main(patterns: bool = False):

    # Initialize recognizer with target size
    recognizer = EigenfaceRecognizer(threshold_0=25, threshold_1=20)
    
    # Train on known faces
    print("Starting training...")
    recognizer.train('./faces')
    
    # Test images
    test_images = [
        'testfaces/1.jpg',
        'testfaces/3.jpg',
        'testfaces/5.jpg',
        'testfaces/11.jpg',
        'testfaces/25.jpg',
        'testfaces/38.jpg',  # Known face
        'testfaces/noface.jpg',  # Not a face
        'testfaces/U1.jpg',  # Unknown face
        'testfaces/U2.jpg',  # Unknown face
        'testfaces/U3.jpg'   # Known face
    ]

    # Test classification
    print("\nTesting classification...")
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Warning: Test image {image_path} not found")
            continue
            
        try:
            result, epsilon_f, min_distance = recognizer.classify_image(image_path)
            print(f"\nImage: {image_path}")
            print(f"Classification: {result}")
            print(f"Distance from face space (epsilon_f): {epsilon_f:.4f}")
            if min_distance is not None:
                print(f"Minimum distance to known faces: {min_distance:.4f}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    # Visualize first few eigenfaces
    if patterns:

        basis_dir = Path('./basis')
        basis_dir.mkdir(exist_ok=True)

        print("\nGenerating eigenface visualizations...")
        for i in range(min(6, recognizer.eigenfaces.shape[1])):
            try:
                output_path = basis_dir / f'eigenface_{i+1}.jpg'    
                recognizer.visualize_eigenface(i, output_path)
                print(f"Saved at ./basis/eigenface_{i+1}.jpg")
            except Exception as e:
                print(f"Error saving eigenface {i+1}: {str(e)}")

if __name__ == "__main__":
    main(patterns = True)