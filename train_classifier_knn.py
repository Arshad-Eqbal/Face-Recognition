import pickle
from sklearn.neighbors import KNeighborsClassifier

# Configuration
EMBEDDINGS_FILE = "embeddings_facenet.pkl"
MODEL_FILE = "knn_model.pkl"
N_NEIGHBORS = 5


def train_classifier():
    """
    Train a K-Nearest Neighbors classifier on the generated face embeddings.
    KNN works well for face recognition as similar faces cluster in embedding space.
    Uses distance-weighted voting to give closer neighbors more influence.
    """
    print("Loading embeddings...")

    # Load the embeddings generated in the previous step
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {EMBEDDINGS_FILE} not found.")
        print("Please run embeddings.py first to generate embeddings.")
        return

    # Extract features and labels
    X = data["embeddings"]
    y = data["labels"]
    class_names = data["class_names"]

    print(f"Training on {len(X)} samples across {len(class_names)} classes...")

    # Initialize KNN classifier with distance weighting
    # Closer neighbors get more weight in the voting process
    knn = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        metric="euclidean",
        weights="distance",
    )

    # Train the classifier
    print(f"Training KNN classifier with {N_NEIGHBORS} neighbors...")
    knn.fit(X, y)

    # Save the trained model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(knn, f)

    print(f"✓ KNN classifier trained and saved to {MODEL_FILE}")
    print(f"✓ Ready for real-time recognition")


if __name__ == "__main__":
    train_classifier()
