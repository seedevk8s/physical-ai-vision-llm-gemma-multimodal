import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def get_hand_points(hand, frame):
    h, w = frame.shape[:2]
    # landmark normalized 좌표를 화면 픽셀 좌표로 변환
    return [(int(p.x * w), int(p.y * h)) for p in hand]


def count_fingers(points, hand_label):
    # 손가락 끝과 손가락 중간 관절 인덱스
    tip_ids = (4, 8, 12, 16, 20)
    pip_ids = (3, 6, 10, 14, 18)
    status = []

    for tip_id, pip_id in zip(tip_ids, pip_ids):
        tip = points[tip_id]
        pip = points[pip_id]

        if tip_id == 4:
            # 엄지손가락은 왼손/오른손 방향에 따라 x 비교
            if hand_label == "Right":
                status.append(tip[0] < pip[0])
            else:
                status.append(tip[0] > pip[0])
        else:
            # 나머지 손가락은 y 좌표로 펴짐 여부 판단
            status.append(tip[1] < pip[1])

    return status


def get_gesture_name(finger_status):
    count = sum(finger_status)
    if count == 0:
        return "Fist"
    if count == 5:
        return "Open palm"
    if finger_status[1] and not any([finger_status[i] for i in [0, 2, 3, 4]]):
        return "Point"
    if finger_status[0] and not any(finger_status[1:]):
        return "Thumbs"
    return f"{count} fingers"


base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보
finger_tips = (4, 8, 12, 16, 20)                                                                # 손가락 끝 landmark의 index

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 프레임 내 손 탐지
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    # 화면 좌측 상단에 텍스트 생성 (손 개수, 왼손/오른손/양손 여부)
    labels = ["Left" if h[0].category_name == "Right" else "Right" for h in result.handedness]
    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, " / ".join(labels), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # 탐지 결과의 각 손마다 선과 점 그리기
    for hand, label in zip(result.hand_landmarks, labels):
        points = get_hand_points(hand, frame)

        # 손 skeleton 그리기
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0,255,0), 2)

        # landmark 지점 표시
        for i, point in enumerate(points):
            color = (0,0,255) if i in finger_tips else (255,0,0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)

        # 손가락 펴짐 상태와 제스처 이름 계산
        finger_status = count_fingers(points, label)
        gesture_name = get_gesture_name(finger_status)
        finger_count = sum(finger_status)

        # 손 영역을 감싸는 사각형 계산
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)

        cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255,255,0), 2)
        cv2.putText(frame, f"{label}: {gesture_name}", (x_min, max(y_min - 15, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Fingers: {finger_count}", (x_min, y_max + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("MediaPipe Hand Detection", frame)

    # q 누르면 종료, s 누르면 현재 화면 저장
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        filename = f"hand_capture_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved screenshot: {filename}")

hand_detector.close()
cap.release()
cv2.destroyAllWindows()