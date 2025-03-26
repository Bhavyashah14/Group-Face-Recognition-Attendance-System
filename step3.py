import cv2
from mtcnn import MTCNN
import numpy as np
from deepface import DeepFace
import os
import csv
from datetime import datetime

# Initialize MTCNN detector
detector = MTCNN()

# Load stored embeddings as a python dictionary
face_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()

# Load the image

#img = cv2.imread('vb.jpeg')
#img = cv2.imread('3bench.jpeg')
img = cv2.imread('unknownfaces.jpeg')

# Resize the image
#img_resized = cv2.resize(img, (600, 600)) #vb
img_resized = cv2.resize(img, (700, 600)) #unknown
#img_resized = cv2.resize(img, (700, 700)) #3 Bench

# Detect faces in the resized image
detected_faces_op = detector.detect_faces(img_resized)

# CSV file setup
csv_filename = "attendance_log.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date"])

# Attendance record
attendance_log = set()

unknown_faces = []  # To store unknown faces and their positions
unknown_counter = 1  # Counter for unknown faces

# Process faces in the image
for i in detected_faces_op:
    # Extract bounding box
    x, y, width, height = i['box']
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x + width, img_resized.shape[1]), min(y + height, img_resized.shape[0])

    # Crop face from the resized image
    face_img = img_resized[y:y2, x:x2]

    # Perform face recognition
    try:
        input_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
        
        if input_embedding:
            input_embedding = np.array(input_embedding[0]['embedding'])

            # Compare with stored embeddings
            min_distance = float("inf")
            recognized_person = "Unknown"
            threshold = 12  

            for name, stored_embedding in face_embeddings.items():
                distance = np.linalg.norm(stored_embedding - input_embedding)
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    recognized_person = name

            if recognized_person != "Unknown":
                attendance_log.add(recognized_person)
                print(f"Recognized: {recognized_person} (Distance: {min_distance:.2f})")
                # Store recognized person's attendance
                with open(csv_filename, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([recognized_person, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            else:
                # Handle unknown faces with unique labels
                recognized_person = f"Unknown{unknown_counter}"
                unknown_faces.append((recognized_person, face_img, x, y, x2, y2))  # Store unknown faces and their positions
                unknown_counter += 1
                print(f"Face not recognized. Labeling as {recognized_person}")

            # Displaying name on image
            cv2.putText(img_resized, recognized_person, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            print("Face not recognized.")

    except Exception as e:
        print(f"Error processing face: {e}")

    # Draw rectangle around face
    cv2.rectangle(img_resized, (x, y), (x2, y2), (255, 0, 0), 3)

# Show the image with all faces (known and unknown)
cv2.imshow('Recognized Faces', img_resized)
cv2.waitKey(0)  # Wait for key press to proceed to registration

# Register all unknown faces
for unknown_name, face_img, x, y, x2, y2 in unknown_faces:
    # Ask for registration name for each unknown face
    name = input(f"Enter name for {unknown_name}: ")
    new_person_dir = os.path.join("known_faces", name)
    os.makedirs(new_person_dir, exist_ok=True)

    # Save the unknown face in the new person's folder
    new_face_path = os.path.join(new_person_dir, f"{name}.jpg")
    cv2.imwrite(new_face_path, face_img)

    # Generate embedding for the new face
    new_embedding = DeepFace.represent(new_face_path, model_name="Facenet", enforce_detection=False)
    if new_embedding:
        face_embeddings[name] = np.array(new_embedding[0]['embedding'])
        np.save("face_embeddings.npy", face_embeddings)  # Save updated embeddings
        print(f"New face registered as {name}!")

cv2.destroyAllWindows()

# Calculate accuracy
total_faces = len(detected_faces_op)
unknown_count = len(unknown_faces)
accuracy = ((total_faces - unknown_count) / total_faces) * 100 if total_faces > 0 else 0

print(f"Recognition Accuracy: {accuracy:.2f}%")
print("Attendance saved in attendance_log.csv")
