import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def ndarray_to_png(arr, filename):
    """Convert ndarray to PNG and save."""
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(filename)


def remove_blue_from_image(image_path, output_path):
    """Remove blue shades from the extracted clothing."""
    
    # Load the extracted clothing image
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for blue color
    lower_blue = np.array([90, 50, 50])  # Lower bound of blue
    upper_blue = np.array([140, 255, 255])  # Upper bound of blue

    # Create a mask to filter out blue pixels
    blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # Invert mask to keep non-blue pixels
    non_blue_mask = cv2.bitwise_not(blue_mask)

    # Apply mask to the image
    result_image = cv2.bitwise_and(image, image, mask=non_blue_mask)

    # Save the cleaned image
    ndarray_to_png(result_image, output_path)
    print(f"✅ Blue pixels removed and cleaned clothing saved as '{output_path}'")

    # Show the final cleaned image
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Cleaned Extracted Clothing")
    plt.axis('off')
    plt.show()


def process_image_pose(image_path):
    """Process the image for pose estimation and clothing extraction."""
    
    # Initialize Mediapipe Pose and Drawing utils
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Pose models for image processing
    pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Read and prepare the image
    image_pose = cv2.imread(image_path)
    image_height, image_width, _ = image_pose.shape
    original_image = image_pose.copy()

    # Convert BGR to RGB for processing
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)

    # Process the image with Pose
    results = pose_image.process(image_in_RGB)

    if not results.pose_landmarks:
        print("❌ No pose landmarks detected.")
        return

    # Extract important points
    MOUTH_RIGHT_X = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x * image_width
    MOUTH_RIGHT_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y * image_height

    LEFT_ANKLE_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height
    RIGHT_ANKLE_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height

    LEFT_ELBOW_X = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
    LEFT_ELBOW_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height

    RIGHT_ELBOW_X = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
    RIGHT_ELBOW_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height

    # Define clothing extraction box with padding for better results
    padding_x = 20  # Add padding to width
    padding_y = 50  # Add padding to height

    STARTING_X = max(int(RIGHT_ELBOW_X) - padding_x, 0)
    STARTING_Y = max(int(MOUTH_RIGHT_Y) - padding_y, 0)
    STARTING_WIDTH = min(int(LEFT_ELBOW_X) - int(RIGHT_ELBOW_X) + 2 * padding_x, image_width - STARTING_X)
    STARTING_HEIGHT = min(int(max(RIGHT_ANKLE_Y, LEFT_ANKLE_Y)) - int(STARTING_Y) + padding_y, image_height - STARTING_Y)

    print(f"Bounding Box: X={STARTING_X}, Y={STARTING_Y}, Width={STARTING_WIDTH}, Height={STARTING_HEIGHT}")

    # Draw landmarks and rectangles on the original image
    mp_drawing.draw_landmarks(
        image=original_image,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    )

    cv2.rectangle(
        original_image, 
        (STARTING_X, STARTING_Y), 
        (STARTING_X + STARTING_WIDTH, STARTING_Y + STARTING_HEIGHT), 
        (0, 0, 255), 2
    )

    # Show pose-detected image
    plt.figure(figsize=[12, 12])
    plt.imshow(original_image[:, :, ::-1])
    plt.title("Pose detected Image with Bounding Box")
    plt.axis('off')
    plt.show()

    # GrabCut for clothing extraction with refined mask
    mask = np.zeros(image_pose.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut with rect initialization
    rect = (STARTING_X, STARTING_Y, STARTING_WIDTH, STARTING_HEIGHT)
    cv2.grabCut(image_pose, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # Refine the mask for probable foreground and sure foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Morphological operations to refine mask
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply the refined mask to extract clothing
    extracted_clothing = image_pose * mask2[:, :, np.newaxis]

    # Save extracted clothing
    ndarray_to_png(extracted_clothing, "extracted_clothing.png")
    print("✅ Extracted clothing saved as 'extracted_clothing.png'")

    # Show extracted clothing
    plt.imshow(cv2.cvtColor(extracted_clothing, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Clothing")
    plt.axis('off')
    plt.show()

    # Remove blue shades after extraction
    remove_blue_from_image("extracted_clothing.png", "extracted_clothing.png")


# Call the function with the input image path
process_image_pose("./output_image.png")
