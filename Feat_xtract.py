import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from rembg import remove
from PIL import Image
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import io

# Load pre-trained models
rcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
rcnn_model.eval()
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


def remove_background(image_bytes):
    """Remove background from the input image."""
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output_image = remove(input_image)  # Remove background

    # Convert RGBA to RGB to match Mask R-CNN expectations
    if output_image.mode == "RGBA":
        output_image = output_image.convert("RGB")
    
    return output_image


def process_image(no_bg_image):
    """Extract clothing items from background-removed image."""
    # Convert to tensor for Mask R-CNN
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(no_bg_image).unsqueeze(0)

    # Detect objects
    with torch.no_grad():
        predictions = rcnn_model(image_tensor)

    clothing_images = []
    for i in range(len(predictions[0]["masks"])):
        if predictions[0]["labels"][i] in [1, 2]:  # 1 = Person, 2 = Clothing
            mask = predictions[0]["masks"][i, 0].detach().numpy()
            mask[mask >= 0.5] = 255
            mask[mask < 0.5] = 0
            mask_image = Image.fromarray(mask).convert("L")

            # Apply mask to original image
            masked_image = Image.composite(
                no_bg_image, Image.new("RGBA", no_bg_image.size, (0, 0, 0, 0)), mask_image
            )
            clothing_images.append(masked_image)

    return clothing_images


def extract_features(img):
    """Extract feature vectors using ResNet50."""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features.flatten()


def extract_features_from_image(image_bytes, save_path="features.npy"):
    """Process image, remove background, and extract feature vector."""
    # Step 1: Remove background
    no_bg_image = remove_background(image_bytes)

    # Step 2: Detect and extract clothing items
    clothing_images = process_image(no_bg_image)

    # Step 3: Extract features from the first detected clothing item
    if not clothing_images:
        return None  # No clothing detected

    # Step 4: Extract features using the background-removed image
    features = extract_features(no_bg_image)

    # Step 5: Save features to a .npy file
    np.save(save_path, features)
    print(f"Features saved successfully to {save_path}")
    return features

# Read image as bytes
with open("extracted_clothing.png", "rb") as img_file:
    image_bytes = img_file.read()

# Extract features and save as features.npy
features = extract_features_from_image(image_bytes)
