import cv2
import numpy as np
import os
import pandas as pd
import ast
from retinaface import RetinaFace  # Assuming you have the RetinaFace library installed

# Function to extract objects and faces
def extract_and_save_objects_and_faces(dataset, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(dataset)):
        # Load the frame 
        frame = os.path.join('obj_det', f'frame{i}.jpg')
        image = cv2.imread(frame)

        boxes = obj_det['boxes'][i]
        boxes = ast.literal_eval(boxes)
        boxes = np.array(boxes)

        # Initialize the object counter
        obj_count = 0

        for j, box in enumerate(boxes):
            # Extract object coordinates
            x, y, width, height = box
            width = width - x
            height = height - y

            # Crop the object
            obj = image[int(y):int(y+height), int(x):int(x+width)]
            if obj.shape[0] == 0 or obj.shape[1] == 0:
                continue

            # make directory for each frame
            os.makedirs(os.path.join(output_folder, f'frame_{i}'), exist_ok=True)

            # Perform face detection on the object
            faces_info = RetinaFace.detect_faces(obj)
            # print size of tuple face_info
            # print(faces_info)
            if type(faces_info) == tuple:
                continue

            for k, box_info in enumerate(faces_info):
                # Extract the face bounding box coordinates
                # print(k, box_info)
                face_bbox = faces_info['face_1']['facial_area']
                x, y, width, height = face_bbox
                width = width - x
                height = height - y

                # Crop and save the face
                face = obj[int(y):int(y+height), int(x):int(x+width)]
                output_path = os.path.join(output_folder, f'frame_{i}', f'object_{j}_face_{k}.jpg')
                cv2.imwrite(output_path, face)

                # Extract and save facial landmarks, facial area as CSV
                # landmarks = faces_info['face_1']['landmarks']
                # landmarks should be co-ordinates of face, with respect to whole frame, not just object, so update face-bbox
                minx_face_bbox = box[0]
                miny_face_bbox = box[1]
                landmarks = face_bbox
                landmarks_df = pd.DataFrame(landmarks).T
                landmarks_df[0] += minx_face_bbox
                landmarks_df[1] += miny_face_bbox
                landmarks_df[2] += minx_face_bbox
                landmarks_df[3] += miny_face_bbox
                landmarks_csv_path = os.path.join(output_folder, f'frame_{i}', f'object_{j}_face_{k}_landmarks.csv')
                landmarks_df.to_csv(landmarks_csv_path, index=False)

# Example usage with the provided dataset
obj_det = pd.read_csv('obj_det.csv')  # Load the dataset

output_folder = 'objects_and_faces_detected'  # Define your output folder

extract_and_save_objects_and_faces(obj_det, output_folder)
