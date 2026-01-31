import cv2
import numpy as np
import pickle
from deepface import DeepFace

# Runtime configuration
CONFIDENCE_THRESHOLD = 0.60
DETECTION_INTERVAL = 5  # Process every Nth frame for performance
MODEL_NAME = "Facenet"  # Facenet for 128D embeddings
DETECTOR_BACKEND = "mtcnn"
EMBEDDINGS_FILE = "embeddings_facenet.pkl"
KNN_MODEL_FILE = "knn_model.pkl"


def load_resources():
    """
    Load the trained KNN classifier and class labels from disk.
    Returns None values if loading fails, which will stop the webcam from running.
    """
    try:
        # Load the trained KNN model
        with open(KNN_MODEL_FILE, "rb") as f:
            knn = pickle.load(f)

        # Load class names for label decoding
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            class_names = data["class_names"]

        print(f"✓ Loaded classifier with {len(class_names)} classes")
        return knn, class_names

    except Exception as e:
        print(f"Error loading resources: {e}")
        print("Please ensure you've run embeddings.py and train_classifier.py first.")
        return None, None


def run_webcam():
    """
    Perform real-time face recognition using the webcam feed.
    Detects faces using MTCNN, generates embeddings with Facenet,
    and classifies them using the trained KNN model.
    """
    # Load classifier and metadata
    knn, class_names = load_resources()
    if knn is None:
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    frame_count = 0
    last_results = []  # Cache results for smooth display

    print("Starting real-time face recognition...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Process faces every DETECTION_INTERVAL frames for better performance
        if frame_count % DETECTION_INTERVAL == 0:
            last_results.clear()

            try:
                # Detect and align faces using MTCNN
                detected_faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                )

                # Process each detected face
                for face_info in detected_faces:
                    # Skip low-confidence detections
                    if face_info["confidence"] < 0.5:
                        continue

                    # Extract bounding box coordinates
                    area = face_info["facial_area"]
                    box = [area["x"], area["y"], area["w"], area["h"]]

                    # Convert face to uint8 for embedding generation
                    face_img = (face_info["face"] * 255).astype(np.uint8)

                    # Generate embedding using Facenet (128D)
                    emb_result = DeepFace.represent(
                        img_path=face_img,
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        align=False,  # Already aligned by extract_faces
                    )

                    if not emb_result:
                        continue

                    # Extract the embedding vector
                    embedding = np.asarray(
                        [emb_result[0]["embedding"]], 
                        dtype=np.float32
                    )

                    # Predict identity using KNN
                    predicted_label = knn.predict(embedding)[0]
                    predicted_name = class_names[predicted_label]

                    # Get confidence score from probability distribution
                    probabilities = knn.predict_proba(embedding)[0]
                    confidence = probabilities[predicted_label]

                    # Determine if confidence meets threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        label = f"{predicted_name} ({confidence * 100:.1f}%)"
                        color = (0, 255, 0)  # Green for recognized
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)  # Red for unknown

                    # Store result for display
                    last_results.append({
                        "box": box,
                        "label": label,
                        "color": color,
                    })

            except Exception as e:
                # Continue processing even if this frame fails
                print(f"Frame processing error: {e}")

        # Draw all cached results on the current frame
        for result in last_results:
            x, y, w, h = result["box"]
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                result["color"],
                2,
            )
            
            # Draw label above the box
            cv2.putText(
                frame,
                result["label"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                result["color"],
                2,
            )

        # Display the frame
        cv2.imshow("Real-Time Face Recognition", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")


if __name__ == "__main__":
    run_webcam()
