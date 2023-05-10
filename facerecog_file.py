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


path1 = "./test/"
test_images = os.listdir(path1)
for i in test_images:
    test_image = f"./test/{i}"
    image = cv2.imread(test_image)
    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""

        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]
        else:
            name = "Stranger"
        
        precision_percentage = 100 - (face_distances[best_match]* 100)

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"{name} ({precision_percentage:.2f}%)", (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)


    # cv2.imshow("Result", image)
    cv2.imwrite(f"./output/{i}_output.jpg", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # break