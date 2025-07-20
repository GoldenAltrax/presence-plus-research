import cv2
import face_recognition
import pickle
import time

# Load encoded faces
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Initialize webcam
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Prepare variables
frame_count = 0
process_every_n = 2  # Process every 2nd frame
prev_time = time.time()

print("[INFO] Starting real-time recognition... Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("[!] Failed to grab frame.")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every Nth frame
    if frame_count % process_every_n == 0:
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        face_names = []

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)
            name = "Unknown"

            if True in matches:
                best_match_index = matches.index(True)
                name = data["names"][best_match_index]

            face_names.append(name)
            print(f"[Detected] {name}")

    # Display results (scale coordinates back up)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show FPS in top left
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Presence+ | Real-Time Face Recognition", frame)

    frame_count += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
