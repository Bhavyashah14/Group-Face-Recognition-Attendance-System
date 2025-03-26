import numpy as np
from deepface import DeepFace
import os

# Directory that is containing known faces
KNOWN_FACES_DIR = "known_faces"

# Dictionary to store face embeddings
face_embeddings = {}

# Iterate through each person's folder
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)

    if os.path.isdir(person_path):  # Ensure it's a directory
        embeddings = []
        print(f"Processing images for {person_name}...")

        # Iterating through all images in person's folder
        for image_file in os.listdir(person_path):
            img_path = os.path.join(person_path, image_file)

            try:
                # Extract face embedding using DeepFace
                embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)

                # Embedding converted to NumPy array and added to embedding list 
                if embedding:
                    embeddings.append(np.array(embedding[0]['embedding']))

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Store the a embedding for this person
        if embeddings:
            face_embeddings[person_name] = np.mean(embeddings, axis=0)
            print(f"Saved embedding for {person_name}.")

# Save embeddings to a .npy file
np.save("face_embeddings.npy", face_embeddings)
print("All face embeddings saved successfully.")
