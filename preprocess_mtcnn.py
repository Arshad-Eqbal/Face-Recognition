import os
import cv2
import numpy as np
from deepface import DeepFace

# Dataset configuration
INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"
DETECTOR_BACKEND = "mtcnn"


def save_original_and_flipped(face_img, output_dir, base_name, index):
    """
    Save both the original detected face and its horizontal mirror image.
    This doubles our training data through simple augmentation.
    """
    original_path = os.path.join(output_dir, f"{base_name}_{index}_orig.jpg")
    flipped_path = os.path.join(output_dir, f"{base_name}_{index}_flip.jpg")

    # Save the original face
    cv2.imwrite(original_path, face_img)

    # Create and save horizontally flipped version
    flipped = cv2.flip(face_img, 1)
    cv2.imwrite(flipped_path, flipped)


def process_dataset():
    """
    Process all images in the dataset by detecting faces using MTCNN,
    aligning them, and saving both original and flipped versions.
    """
    # Verify input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: '{INPUT_DIR}' directory not found.")
        return

    print(f"Using {DETECTOR_BACKEND} for face detection and alignment...")

    # Process each person's folder in the dataset
    for person_name in os.listdir(INPUT_DIR):
        person_path = os.path.join(INPUT_DIR, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue

        # Create output directory for this person
        output_person_dir = os.path.join(OUTPUT_DIR, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        source_count = 0

        # Process each image for this person
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                # Detect and align face using MTCNN
                detected_faces = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=DETECTOR_BACKEND,
                    align=True
                )

                # Skip if no face was detected
                if not detected_faces:
                    print(f"Skipping {img_path}: no face detected.")
                    continue

                # Use the first detected face (highest confidence)
                face_rgb = detected_faces[0]["face"]

                # Convert from normalized RGB [0,1] to BGR uint8 [0,255]
                face_uint8 = (face_rgb * 255).astype(np.uint8)
                face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

                # Save both original and flipped versions
                save_original_and_flipped(
                    face_bgr,
                    output_person_dir,
                    person_name,
                    source_count
                )

                source_count += 1

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        print(
            f"Processed {person_name}: "
            f"{source_count} source images → {source_count * 2} total images."
        )


if __name__ == "__main__":
    process_dataset()
    print("Done. Preprocessed dataset ready for embedding generation.")
