from remove_bg import remove_background
from R_CNN import process_image_pose, remove_blue_from_image
from Feat_xtract import extract_features
from similarity_score import find_top_matches
import json
import numpy as np


# --- Load Vectors from Data.js ---
def load_vectors(data_path="data.js"):
    """Load stored vectors from data.js."""
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


# --- Extract Vectors and Product Info ---
def extract_vectors_and_info(stored_data):
    """Extract feature vectors and product information from stored data."""
    vectors = []
    product_info = []
    
    for item in stored_data:
        vectors.append(np.array(item["vector"]))  # Extract vector
        product_info.append({"product_id": item["product_id"], "image": item["image"]})
    
    return np.array(vectors), product_info


# --- Main Workflow ---
def process_and_match(input_path, data_path="data.js", top_n=5):
    """Complete pipeline: remove background, extract clothing, features, and match."""
    print("🚀 Starting Process...")

    # Step 1: Remove Background
    no_bg_image = remove_background(input_path)
    no_bg_image.save("output_image.png")

    # Step 2: Extract Clothing using Pose Detection
    extracted_path = process_image_pose("output_image.png")
    if extracted_path is None:
        print("❌ No clothing extracted. Exiting...")
        return

    # Step 3: Clean Blue Shades
    remove_blue_from_image(extracted_path, "extracted_clothing.png")

    # Step 4: Extract Feature Vector
    feature_vector = extract_features("extracted_clothing.png")

    # Step 5: Load Vectors from Data.js and Extract Vectors/Info
    stored_data = load_vectors(data_path)
    db_vectors, product_info = extract_vectors_and_info(stored_data)

    # Step 6: Find Top Matches and Similarity Scores
    top_matches, similarity_scores = find_top_matches(feature_vector, db_vectors, top_n)

    # Print Top Matches with Similarity Scores
    for i, (match_idx, score) in enumerate(zip(top_matches, similarity_scores), start=1):
        match_info = product_info[match_idx]
        print(f"🔍 Match {i}: Product ID: {match_info['product_id']}, Image: {match_info['image']}, Similarity Score: {score:.4f}")

    return top_matches, similarity_scores


# --- Run Pipeline ---
if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Input image path
    process_and_match(input_image_path)
