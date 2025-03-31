import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# --- Load All .npy Files from Folder ---
def load_db_features_from_folder(folder_path="data"):
    """Load all .npy feature vectors from a folder."""
    feature_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    
    db_features = []
    file_names = []

    for file_name in feature_files:
        file_path = os.path.join(folder_path, file_name)
        
        # Load feature vectors
        try:
            features = np.load(file_path)
            db_features.append(features)
            file_names.append(file_name)
        except Exception as e:
            print(f"⚠️ Error loading {file_name}: {e}")
    
    if not db_features:
        print("❌ No valid feature files found in the folder.")
        return None, None

    # Convert list to numpy array
    db_features = np.vstack(db_features)
    print(f"✅ Loaded {len(db_features)} feature vectors from '{folder_path}'")

    return db_features, file_names


# --- Compute Cosine Similarity and Find Top Matches ---
def find_top_matches(vector_path="features.npy", folder_path="data", top_n=5):
    """Compare feature vector with all feature vectors in 'data' folder and return top matches."""
    
    # Load feature vector from the input file
    try:
        input_vector = np.load(vector_path)
    except Exception as e:
        print(f"❌ Error loading {vector_path}: {e}")
        return

    # Reshape vector if necessary
    if input_vector.ndim == 1:
        input_vector = input_vector.reshape(1, -1)

    # Load all database features
    db_features, file_names = load_db_features_from_folder(folder_path)
    if db_features is None or file_names is None:
        return

    # Compute cosine similarity
    print("🔍 Calculating similarity scores...")
    similarity_scores = cosine_similarity(input_vector, db_features)[0]

    # Get indices of top matches
    top_matches_idx = np.argsort(similarity_scores)[::-1][:top_n]

    # Display Top Matches
    print("🎯 Top Matches:")
    for i, idx in enumerate(top_matches_idx, start=1):
        print(f"🔹 Match {i}: File: {file_names[idx]}, Similarity Score: {similarity_scores[idx]:.4f}")

    return top_matches_idx, [similarity_scores[idx] for idx in top_matches_idx]


# --- Run Pipeline ---
if __name__ == "__main__":
    # Path to the feature vector to compare
    vector_path = "features.npy"  # Path to the extracted feature vector
    folder_path = "data"          # Folder where all feature vectors are stored

    # Find top matches
    find_top_matches(vector_path, folder_path)
