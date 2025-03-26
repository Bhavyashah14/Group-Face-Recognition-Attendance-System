import cv2
import os
from mtcnn import MTCNN

# Create directory to store known faces
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize webcam to collect images of users
cap = cv2.VideoCapture(0)
detector = MTCNN()

# Enter number of users to create dataset
num_people = int(input("Enter the number of people to collect images for: "))

for person in range(num_people):
    # Enter name for each user and create directory
    name = input(f"Enter name for person {person + 1}: ")
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0 # Keeping count of image
    print(f"Capturing images for {name}...")

    # Capture 10 images per person for dataset creation
    while count < 10:
        # Reading frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access webcam.")
            break

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        for face in faces:
            # Extracts bounding box
            x, y, width, height = face['box']
            # x,y : Top Left Corner
            x, y = max(x, 0), max(y, 0)
            # x2,y2 : Bottom Right Corner
            x2, y2 = x + width, y + height

            # Crop and save the face image
            face_img = frame[y:y2, x:x2]
            file_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            count += 1

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Capturing Face Images", frame)

        # Stop capturing if 'q' is pressed or 10 images are taken
        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 10:
            break

    print(f"Saved 10 images for {name}.")

# Release resources (Camera, Window)
cap.release()
cv2.destroyAllWindows()
print("Image collection complete.")
print("Dataset collected!")