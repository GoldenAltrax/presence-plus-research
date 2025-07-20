import face_recognition
import os
import pickle
from collections import defaultdict
import numpy as np

face_dir = "face_db"
known_faces = defaultdict(list)  # Dictionary: {name: [enc1, enc2, ...]}

# STEP 1: Load and encode all images
for file in os.listdir(face_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(file)[0]
        # Strip numbers like "ibrahim1" → "ibrahim"
        name = ''.join(filter(str.isalpha, name))

        image = face_recognition.load_image_file(f"{face_dir}/{file}")
        enc = face_recognition.face_encodings(image)

        if enc:
            known_faces[name].append(enc[0])
            print(f"[✓] Encoded face for {name} from {file}")
        else:
            print(f"[!] No face found in {file}")

# STEP 2: Average encodings per name
encodings = []
names = []

for name, enc_list in known_faces.items():
    avg_enc = np.mean(enc_list, axis=0)  # Average all encodings for this person
    encodings.append(avg_enc)
    names.append(name)

# STEP 3: Save to file
data = {"encodings": encodings, "names": names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print(f"[✓] Saved averaged encodings for {len(names)} unique faces.")
