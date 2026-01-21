import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder

# Configuration
INPUT_DIR = "processed_dataset"
MODEL_NAME = "Facenet"  # Facenet generates 128-dimensional embeddings
OUTPUT_FILE = "embeddings_facenet.pkl"


def generate_embeddings():
    """
    Generate face embeddings using Facenet for all processed images.
    Facenet produces 128-dimensional embeddings ideal for face recognition.
    Saves embeddings along with their labels for training the classifier.
    """
    print(f"Generating embeddings using {MODEL_NAME}...")

    embeddings = []
    labels = []

    # Process each person's folder
    for person_name in os.listdir(INPUT_DIR):
        person_dir = os.path.join(INPUT_DIR, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue

        # Process each image for this person
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            try:
                # Generate embedding using Facenet (128D)
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                )

                # Extract the embedding vector and add to our dataset
                embeddings.append(result[0]["embedding"])
                labels.append(person_name)

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    # Check if we generated any embeddings
    if not embeddings:
        print("Error: No embeddings generated. Check your dataset.")
        return

    # Convert to numpy arrays for sklearn compatibility
    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels)

    print(f"Generated {len(embeddings)} embeddings with shape: {X.shape}")

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Package everything for saving
    output_data = {
        "embeddings": X,
        "labels": y_encoded,
        "class_names": label_encoder.classes_,
    }

    # Save to disk
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output_data, f)

    print(f"✓ Saved {len(embeddings)} embeddings to {OUTPUT_FILE}")
    print(f"✓ Classes: {list(label_encoder.classes_)}")


if __name__ == "__main__":
    generate_embeddings()
