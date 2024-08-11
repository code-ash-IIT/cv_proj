import pandas as pd
import os
from deepface import DeepFace

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

faces_inp = 'enhanced_faces_swinir'
landmarks_for_csv_dir = 'objects_and_faces_detected'
criminal_img_path = 'criminal.png'

# Create a dictionary to store the closest match per frame
closest_matches = {}

# Iterate over all the frames in enhanced_faces_swinir
for frame_folder in os.listdir(faces_inp):
    for file in os.listdir(os.path.join(faces_inp, frame_folder)):
        # If not a jpg, continue
        if not file.endswith('.jpg'):
            continue
        inp_file_path = os.path.join(faces_inp, frame_folder, file)

        # Find the similarity score between inp_file_path and criminal_img_path
        result = DeepFace.verify(
            img1_path=inp_file_path,
            img2_path=criminal_img_path,
            model_name=models[6],
            enforce_detection=False,
        )

        if result['verified']:
            print(f'Found a match in {inp_file_path}')
            print(f'Similarity score = {result["distance"]}')

            # Check if there is already a stored match for this frame
            if frame_folder not in closest_matches or result['distance'] < closest_matches[frame_folder]['Similarity_score'].values[0]:
                # Find landmarks for inp_file_path
                landmarks_file_path = os.path.join(landmarks_for_csv_dir, frame_folder, file.replace('.jpg', '_landmarks.csv'))
                landmarks = pd.read_csv(landmarks_file_path)

                # Add a column for file name
                landmarks['file_name'] = file
                landmarks['Similarity_score'] = result['distance']
                closest_matches[frame_folder] = landmarks

# Save the closest match for each frame in the 'verified_faces' folder
os.makedirs('verified_faces', exist_ok=True)
for frame_folder, landmarks in closest_matches.items():
    file_name = f'{frame_folder}.csv'
    landmarks.to_csv(os.path.join('verified_faces', file_name), index=False)
