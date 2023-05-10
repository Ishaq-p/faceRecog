import face_recognition as fr
import cv2
import numpy as np
import os

path = "./train/"

known_names = []
known_name_encodings = []

images = os.listdir(path)

for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encodings = fr.face_encodings(image)
    if len(encodings) > 0:
        encoding = encodings[0]
        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    face_locations = fr.face_locations(frame)

    face_encodings = fr.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding, tolerance=0.6)
        name = ""

        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
        else:
            name = "Stranger"

        precision_percentage = 100 - (face_distances[best_match_index] * 100)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({precision_percentage:.2f}%)", (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 4)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

