from ultralytics import YOLO
import cv2

# load YOLOv8 model
model = YOLO('./best_v6.pt') # best.pt

# load video
video_path = './test2.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    # detect objects # track objects
    results = model.track(frame, persist=True)

    # plot results
    results[0].names[0] = 'shuttle'

    frame_ = results[0].plot()
    print(results[0])

    # visualize
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break