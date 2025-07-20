import cv2
import face_recognition
import pickle
import time
import json
import os
from datetime import datetime
from drive_upload import upload_file_to_drive

# --- Load Encoded Faces ---
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# --- Setup Video ---
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Create recordings folder ---
os.makedirs("recordings", exist_ok=True)

# --- Generate filename based on timestamp ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
video_filename = f"recordings/session_{timestamp}.mp4"

# --- Setup VideoWriter ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

# --- Init Variables ---
prev_time = time.time()
attendance = {}

# --- Session Info ---
session_info = {
    "class_name": "IoT",
    "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

# --- Init Attendance Status ---
for name in data["names"]:
    attendance[name] = {
        "status": "Absent",
        "entry_time": None,
        "exit_time": None,
        "last_seen": 0.0,
        "duration_sec": 0,
        "grace_started": None,
        "countdown": None,
        "timeline": []  # <-- 🆕 add timeline tracking
    }

print("[INFO] Starting Presence+ real-time system. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("[!] Camera read failed.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    face_names = []

    current_time = time.time()
    detected_names = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            best_match_index = matches.index(True)
            name = data["names"][best_match_index]
            detected_names.append(name)

            person = attendance[name]

            if person["status"] != "Present":
                # ⏱ Record timeline entry
                person["timeline"].append({
                    "status": "Present",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            if person["status"] != "Present" and person["entry_time"] is None:
                person["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            person["status"] = "Present"
            person["last_seen"] = current_time
            person["grace_started"] = None
            person["countdown"] = None

        face_names.append(name)
        print(f"[Detected] {name}")

    # Update attendance status
    grace_period = 2
    absent_after = 30

    for person_name, info in attendance.items():
        if info["status"] == "Present" and person_name not in detected_names:
            if info["grace_started"] is None:
                info["grace_started"] = current_time

            grace_elapsed = current_time - info["grace_started"]

            if grace_elapsed >= grace_period:
                info["status"] = "Going Absent"
                info["countdown"] = absent_after
                info["last_seen"] = current_time

        elif info["status"] == "Going Absent":
            countdown_remaining = absent_after - int(current_time - info["last_seen"])
            if countdown_remaining > 0:
                info["countdown"] = countdown_remaining
            else:
                info["status"] = "Absent"
                info["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # ⏱ Record timeline entry
                info["timeline"].append({
                    "status": "Absent",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                if info["entry_time"]:
                    entry_dt = datetime.strptime(info["entry_time"], "%Y-%m-%d %H:%M:%S")
                    info["duration_sec"] += int(current_time - entry_dt.timestamp())

                info["countdown"] = None
                info["grace_started"] = None

        elif info["status"] == "Absent" and person_name in detected_names:
            info["status"] = "Present"

            info["timeline"].append({
                "status": "Present",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            if info["entry_time"] is None:
                info["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            info["last_seen"] = current_time
            info["grace_started"] = None
            info["countdown"] = None

    # Draw face boxes
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display attendance status list
    y = 60
    for person, info in attendance.items():
        status = info["status"]
        countdown = info["countdown"]

        if status == "Present":
            status_text = f"{person}: Present"
            color = (0, 255, 0)

        elif status == "Going Absent":
            status_text = f"{person}: Going Absent in {countdown}s"
            color = (0, 165, 255)

        elif status == "Absent":
            status_text = f"{person}: Absent"
            color = (0, 0, 255)

        else:
            status_text = f"{person}: Unknown"
            color = (200, 200, 200)

        cv2.putText(frame, status_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 30

    cv2.imshow("Presence+ | Attendance Tracker", frame)
    out.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# --- Finalize and Save ---
video.release()
out.release()
cv2.destroyAllWindows()

session_info["attendance"] = attendance
session_info["session_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

start_dt = datetime.strptime(session_info["session_start"], "%Y-%m-%d %H:%M:%S")
end_dt = datetime.strptime(session_info["session_end"], "%Y-%m-%d %H:%M:%S")
session_info["session_duration_sec"] = int((end_dt - start_dt).total_seconds())

attendance_filename = f"attendance_{timestamp}.json"
with open(attendance_filename, "w") as f:
    json.dump(session_info, f, indent=4)

print(f"[✓] Session saved to {attendance_filename} and {video_filename}")

upload_file_to_drive(attendance_filename, folder_id="1F8h8KwLlIhJNjyAm6oM9qMRQkR4PM5y1")
upload_file_to_drive(video_filename, folder_id="1F8h8KwLlIhJNjyAm6oM9qMRQkR4PM5y1")
