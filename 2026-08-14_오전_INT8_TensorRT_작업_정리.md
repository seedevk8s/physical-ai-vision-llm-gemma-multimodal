# 2026-08-14 오전 작업 정리: YOLO11n INT8 TensorRT

## 1. 오늘 오전 작업의 목표

기존 YOLO11n 모델을 NVIDIA Jetson에서 더 효율적으로 실행하기 위해 INT8 TensorRT 엔진을 생성하고, CSI 카메라 영상으로 실시간 객체 탐지를 실행할 수 있는 코드를 준비했다.

쉽게 말하면 다음 과정이다.

> 기존 PyTorch 모델을 INT8 정밀도의 TensorRT 실행 파일로 최적화하고, 실제 카메라 환경에서 속도와 탐지 결과를 확인할 수 있도록 구성했다.

## 2. 전체 작업 흐름

```text
YOLO11n 원본 모델
  yolo11n.pt
        ↓
Calibration 이미지 수집
  Calibration.py
        ↓
Calibration 데이터셋 500장 생성
  calibration.yaml + images + labels
        ↓
INT8 양자화 및 TensorRT 엔진 변환
  INT8_Quantization.py
        ↓
INT8 TensorRT 엔진 생성
  yolo11n_int8.engine
        ↓
CSI 카메라 실시간 객체 탐지
  INT8TensorRT.py
        ↓
탐지 결과와 End-to-End FPS 표시
```

## 3. Calibration 데이터셋 준비

INT8 양자화는 모델의 숫자 표현 범위를 결정하기 위해 실제 입력과 비슷한 대표 이미지가 필요하다. 이를 Calibration이라고 한다.

`Calibration.py`를 이용해 Jetson CSI 카메라에서 영상을 받아 다음 데이터셋을 준비했다.

| 항목 | 내용 |
|---|---|
| 이미지 수 | 500장 |
| 저장 간격 | 5프레임마다 1장 |
| 이미지 경로 | `src/datasets/calibration/images/` |
| 라벨 경로 | `src/datasets/calibration/labels/` |
| 설정 파일 | `src/datasets/calibration/calibration.yaml` |
| 클래스 | COCO 80개 클래스 |

라벨 파일은 각 이미지와 짝을 맞추기 위한 빈 텍스트 파일이다. 이 단계의 목적은 모델을 다시 학습하는 것이 아니라, TensorRT가 INT8 변환에 사용할 입력 분포를 수집하는 것이다.

## 4. INT8 TensorRT 엔진 생성

`INT8_Quantization.py`에서 다음 입력을 사용했다.

```text
원본 모델:       src/models/YOLO/yolo11n.pt
Calibration YAML: src/datasets/calibration/calibration.yaml
출력 엔진:       src/models/YOLO/yolo11n_int8.engine
```

주요 변환 조건은 다음과 같다.

| 설정 | 값 | 의미 |
|---|---:|---|
| 출력 형식 | `engine` | TensorRT 엔진 생성 |
| 입력 크기 | `640` | YOLO 입력 크기 640×640 |
| 정밀도 | `int8=True` | INT8 양자화 사용 |
| Batch | `1` | 한 번에 프레임 한 장 처리 |
| Dynamic shape | `False` | 고정 입력 크기 사용 |
| Device | `0` | 첫 번째 NVIDIA GPU 사용 |
| NMS 포함 | `False` | NMS를 엔진 외부 후처리에서 수행 |

생성된 주요 파일은 다음과 같다.

| 파일 | 크기 | 역할 |
|---|---:|---|
| `yolo11n.pt` | 약 5.4 MB | 원본 PyTorch 모델 |
| `yolo11n_fp16.engine` | 약 8.3 MB | 기존 FP16 TensorRT 엔진 |
| `yolo11n_int8.engine` | 약 4.8 MB | 오늘 생성한 INT8 TensorRT 엔진 |
| `yolo11n.cache` | 약 16 KB | INT8 Calibration 관련 캐시 |
| `yolo11n_int8_raw.engine` | 약 4.8 MB | 추가로 생성되어 보관 중인 INT8 엔진 파일 |

INT8 엔진은 FP16 엔진보다 파일 크기가 작다. 다만 실제 속도 향상과 탐지 정확도 변화는 동일한 환경에서 직접 측정해야 판단할 수 있다.

## 5. 실시간 INT8 객체 탐지 코드 준비

`INT8TensorRT.py`는 양자화를 수행하는 코드가 아니다. 앞 단계에서 생성한 `yolo11n_int8.engine`을 불러와 실제 카메라 영상에 사용하는 실행 단계다.

실행 과정은 다음과 같다.

```text
CSI 카메라
    → GStreamer
    → OpenCV 프레임
    → YOLO11n INT8 TensorRT 추론
    → NMS 및 탐지 결과 그리기
    → FPS 표시
    → 화면 출력
```

카메라와 추론 설정은 다음과 같다.

| 항목 | 설정 |
|---|---|
| 카메라 | CSI 카메라 `sensor-id=0` |
| 카메라 해상도 | 1280×720 |
| 카메라 프레임률 | 30 FPS |
| 모델 입력 엔진 | `yolo11n_int8.engine` |
| Confidence threshold | 0.25 |
| IoU threshold | 0.5 |
| 탐지 클래스 | 전체 클래스 |
| 종료 키 | `q` |

GStreamer의 `queue leaky=downstream`, `max-size-buffers=1`, `drop=true` 설정으로 처리하지 못한 오래된 프레임이 계속 쌓이는 것을 방지한다. 이는 화면 지연을 줄이는 데 도움이 된다.

표시되는 FPS에는 모델 연산뿐 아니라 다음 과정이 함께 포함된다.

- 카메라 프레임 획득
- 이미지 전처리와 GPU 데이터 전달
- TensorRT 추론
- NMS 등의 후처리
- Bounding Box와 텍스트 그리기
- OpenCV 화면 출력

따라서 이 값은 순수 엔진 처리량보다 실제 사용 환경의 체감 성능에 가까운 End-to-End FPS다. 화면의 FPS는 순간적인 흔들림을 줄이기 위해 지수 이동 평균으로 표시한다.

## 6. 콘솔 실행 방법

프로젝트 폴더로 이동한 후 가상환경을 활성화한다.

```bash
cd /home/chjin/vision-llm
source .venv/bin/activate
python INT8TensorRT.py
```

가상환경을 활성화하지 않고 다음과 같이 직접 실행할 수도 있다.

```bash
cd /home/chjin/vision-llm
./.venv/bin/python INT8TensorRT.py
```

현재 확인된 실행 환경은 다음과 같다.

| 구성 요소 | 확인 결과 |
|---|---|
| Python | 3.10.12 |
| OpenCV | 4.8.0 |
| OpenCV GStreamer | 사용 가능 |
| Ultralytics | 8.4.118 |
| 가상환경 | `.venv` 준비 완료 |
| INT8 엔진 | 준비 완료 |

시스템 기본 `python3` 환경에는 현재 `ultralytics`가 없으므로 `.venv`의 Python을 사용해야 한다.

## 7. 실행 시 확인할 항목

- CSI 카메라가 정상적으로 열리는가?
- 객체의 Bounding Box와 클래스가 올바르게 표시되는가?
- 화면 지연이나 끊김이 발생하지 않는가?
- 일정 시간 실행했을 때 평균 FPS가 어느 정도인가?
- FP16 엔진과 비교했을 때 탐지 누락이나 오탐이 증가하지 않는가?
- 장시간 실행 시 GPU 온도와 전력 상태가 안정적인가?

실행 중 다른 터미널에서 Jetson 상태를 확인할 수 있다.

```bash
tegrastats
```

Power Mode도 함께 기록하면 비교 결과의 신뢰도를 높일 수 있다.

```bash
sudo nvpmodel -q
```

## 8. 문제 발생 시 점검 사항

### `ModuleNotFoundError: No module named 'ultralytics'`

시스템 Python이 아니라 프로젝트의 `.venv`를 사용한다.

```bash
./.venv/bin/python INT8TensorRT.py
```

### 프로그램이 바로 종료되는 경우

`cap.read()`가 실패했을 가능성이 있다. CSI 카메라 연결, 카메라 사용 중인 다른 프로세스 및 `sensor-id`를 확인한다.

카메라만 별도로 시험하려면 다음 명령을 사용할 수 있다.

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 \
  ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' \
  ! nvvidconv ! autovideosink
```

### 화면을 열 수 없는 경우

SSH 환경에서 GUI 디스플레이 연결 없이 실행하면 OpenCV 창을 열지 못할 수 있다. Jetson의 데스크톱 터미널에서 실행하거나 디스플레이 전달 설정을 확인한다.

### TensorRT 엔진 호환 오류

TensorRT 엔진은 생성에 사용한 GPU 구조와 TensorRT 버전에 영향을 받는다. 일반적으로 실제 사용할 Jetson 장치에서 엔진을 생성하고 동일한 환경에서 실행하는 것이 안전하다.

## 9. 오늘 오전 완료 사항

- [x] CSI 카메라 기반 Calibration 수집 코드 준비
- [x] Calibration 이미지 500장 수집
- [x] 이미지별 라벨 파일 500개 준비
- [x] `calibration.yaml` 생성
- [x] YOLO11n INT8 TensorRT 변환 코드 준비
- [x] `yolo11n_int8.engine` 생성
- [x] Calibration 캐시 생성
- [x] INT8 엔진 기반 실시간 카메라 추론 코드 준비
- [x] 실행에 필요한 Python 가상환경과 패키지 확인
- [ ] INT8 실시간 추론의 평균 FPS 기록
- [ ] FP16과 INT8의 동일 조건 성능 비교
- [ ] FP16과 INT8의 탐지 품질 비교
- [ ] GPU 사용률·온도·전력 기록

## 10. 다음 작업

동일한 카메라 환경에서 `FPS_TensorRT.py`의 FP16 엔진과 `INT8TensorRT.py`의 INT8 엔진을 각각 실행하고 아래 표를 채운다.

| 모델 | 정밀도 | 평균 FPS | GPU 사용률 | 온도 | 전력 | 탐지 품질 |
|---|---|---:|---:|---:|---:|---|
| `yolo11n_fp16.engine` | FP16 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 기준 |
| `yolo11n_int8.engine` | INT8 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 비교 필요 |

비교할 때는 카메라 해상도, 모델 입력 크기, Confidence, IoU, 화면 출력 여부, Jetson Power Mode 및 온도 조건을 동일하게 유지해야 한다.

## 11. 한 줄 요약

오늘 오전에는 CSI 카메라에서 수집한 500장의 대표 이미지로 YOLO11n INT8 Calibration 환경을 만들고, 약 4.8 MB의 TensorRT INT8 엔진을 생성한 뒤, 이를 실시간 카메라 객체 탐지와 FPS 측정에 사용할 수 있도록 준비했다.
