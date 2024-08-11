from det_executor import DetExecutor, draw_on_image
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

# Start the video capture
cap = cv2.VideoCapture('CCTV_Mall.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# define model
# loading model
name = 'yolov7'
ex = DetExecutor(name)
frames = []
df = pd.DataFrame(columns=['frame', 'class', 'score', 'boxes'])
curr_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    classes, boxes, scores = ex.predict(img)
    out_frame = img
    b=[]; c=[]; s=[]
    for i in range(len(classes[0])):
        if classes[0][i] == 0: #person
            b.append(boxes[0][i])
            c.append(classes[0][i])
            s.append(scores[0][i])
    out_frame = draw_on_image(img, b, s, c)
    frames.append(out_frame)
    df.loc[len(df)] = [curr_frame, c, s, b]
    curr_frame += 1

# define output folder to store frames
output_path = 'obj_det'
os.makedirs(output_path, exist_ok=True)

cap.release()
cv2.destroyAllWindows()

# store the frames
for i in range(len(frames)):
    cv2.imwrite(os.path.join(output_path, f"frame{i}.jpg"), frames[i])

# ------------------
df.to_csv('obj_det.csv', index=False)

# ------------------
# show frames in directory obj_det using opencv, one by one like a video

for i in range(len(frames)):
    img =  cv2.imread(os.path.join(output_path, f"frame{i}.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", img)
    #  dont wait for any key, just show the frame and move to next frame after destroying the previous frame
    cv2.waitKey(1)
    cv2.destroyAllWindows()

cv2.waitKey(1)
cv2.destroyAllWindows()

