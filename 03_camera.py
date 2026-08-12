import cv2


pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("VideoCapture with GStreamer", frame)

cap.release()
cv2.destroyAllWindows()