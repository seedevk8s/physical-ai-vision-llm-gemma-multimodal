# 현재 작업 정리: Jetson에서 YOLO 객체 탐지 최적화

## 한눈에 보기

현재 진행 중인 작업은 **YOLO11n 객체 탐지 모델을 NVIDIA Jetson Orin에서 빠르게 실행하도록 TensorRT FP16 엔진으로 최적화하는 것**이다.

쉽게 말하면 다음과 같다.

> 사람이 학습과 개발에 사용하기 편한 PyTorch 모델을, Jetson GPU가 빠르게 실행하기 좋은 TensorRT 모델로 바꾸고 성능을 확인하는 작업이다.

현재는 **FP16 TensorRT 엔진 생성과 기본 추론 벤치마크까지 성공한 상태**다. 다음 핵심 작업은 실제 카메라 영상에서 `.pt` 모델과 `.engine` 모델의 체감 성능을 비교하는 것이다.

## 전체 작업 흐름

```text
객체 탐지 기초 학습
    ↓
YOLO11n PyTorch 모델 실행 (.pt)
    ↓
ONNX 중간 모델로 변환 (.onnx)
    ↓
TensorRT FP16 엔진 생성 (.engine)  ← 현재 완료
    ↓
trtexec으로 순수 추론 성능 측정   ← 현재 완료
    ↓
실시간 카메라 객체 탐지 적용       ← 다음 확인 단계
    ↓
.pt와 .engine의 실제 FPS 비교
    ↓
필요하면 INT8 양자화로 추가 최적화
```

## 왜 이 작업을 하는가?

Jetson은 카메라와 AI 모델을 현장에서 직접 실행하기 좋은 Edge AI 장치이지만, 데스크톱 GPU보다 연산 성능·메모리·전력에 제약이 있다.

카메라 영상은 계속 새로운 프레임이 들어오기 때문에 모델의 처리 속도가 느리면 다음 문제가 생긴다.

- 화면이 끊기거나 지연된다.
- 객체 위치가 실제 움직임보다 늦게 표시된다.
- 처리하지 못한 프레임이 쌓이거나 버려진다.
- 전력 사용량과 발열이 증가할 수 있다.

따라서 YOLO 모델의 정확도를 가능한 한 유지하면서 추론 시간을 줄이는 최적화가 필요하다. 이번 작업에서는 NVIDIA GPU에 특화된 **TensorRT**와 연산 정밀도를 낮춘 **FP16**을 사용한다.

## 현재 사용 중인 모델 파일

| 파일 | 역할 | 현재 상태 |
|---|---|---|
| `src/models/YOLO/yolo11n.pt` | 원본 PyTorch YOLO11n 모델 | 준비 완료 |
| `src/models/YOLO/yolo11n.onnx` | PyTorch와 TensorRT 사이의 중간 모델 | 변환 완료 |
| `src/models/YOLO/yolo11n_fp16.engine` | Jetson GPU용 FP16 TensorRT 모델 | 생성 완료 |

각 형식의 관계는 다음과 같다.

- `.pt`: PyTorch에서 사용하기 편하지만 배포 환경에서 최상의 속도는 아닐 수 있다.
- `.onnx`: 서로 다른 AI 프레임워크 사이에서 모델을 전달하기 위한 표준 형식이다.
- `.engine`: 현재 Jetson과 TensorRT 환경에 맞게 최적화된 실행 파일이다.

> TensorRT 엔진은 GPU 구조와 TensorRT 버전 등에 영향을 받으므로, 일반적으로 실제 사용할 Jetson 장치에서 생성하고 사용하는 것이 안전하다.

## 지금까지 학습하고 구현한 내용

### 1. 객체 탐지의 기본 원리

`04_DL-Object-Detection.ipynb`에서 객체 탐지의 핵심 개념을 단계적으로 실습했다.

- Bounding Box: 탐지한 객체의 위치를 사각형 좌표로 표현
- Confidence Score: 탐지 결과를 모델이 얼마나 확신하는지 나타내는 값
- IoU: 두 Bounding Box가 얼마나 겹치는지 나타내는 비율
- NMS: 같은 객체에 중복으로 생성된 Bounding Box를 제거하는 과정

색상 분할과 Contour를 이용한 규칙 기반 탐지를 먼저 구현한 뒤, 학습 기반 모델인 YOLO로 확장했다. 이 과정을 통해 YOLO의 출력에 포함된 좌표, 클래스, 신뢰도와 NMS가 어떤 의미인지 확인했다.

### 2. YOLO11n 객체 탐지

COCO 데이터셋의 80개 클래스로 사전 학습된 YOLO11n을 사용했다. `n`은 Nano 모델을 뜻하며, 모델이 작고 빠르기 때문에 Jetson 같은 Edge 장치의 실시간 추론에 적합하다.

정적 이미지에서는 다음 기능을 확인했다.

- 이미지 한 장 및 여러 장에 대한 객체 탐지
- 탐지된 Bounding Box 시각화
- 클래스 인덱스와 Confidence Score 확인
- 특정 클래스만 선택하여 탐지

### 3. 실시간 카메라 처리 구조

CSI 카메라 영상을 GStreamer와 OpenCV로 받아 각 프레임을 YOLO에 입력하는 구조를 준비했다.

```text
CSI 카메라
    → GStreamer
    → OpenCV 프레임
    → YOLO 추론
    → NMS 및 Bounding Box 표시
    → 화면 출력
```

표시되는 FPS는 순간적인 흔들림을 줄이기 위해 EMA(지수 이동 평균) 방식으로 계산한다.

### 4. ONNX 변환 및 검증

원본 `yolo11n.pt` 모델을 고정 입력 크기 `1 × 3 × 640 × 640`, batch 1, opset 20 조건으로 ONNX 형식으로 변환했다.

생성된 모델은 다음 경로에 있다.

```text
src/models/YOLO/yolo11n.onnx
```

TensorRT 로그에서도 ONNX 모델 파싱이 정상적으로 완료되었다.

### 5. TensorRT FP16 엔진 생성

다음 명령으로 ONNX 모델을 FP16 TensorRT 엔진으로 변환했다.

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx="$HOME/vision-llm/src/models/YOLO/yolo11n.onnx" \
    --saveEngine="$HOME/vision-llm/src/models/YOLO/yolo11n_fp16.engine" \
    --fp16
```

명령 실행 결과는 `PASSED`였으며 엔진이 정상 생성되었다.

## 현재까지 확인된 성능

`trtexec`은 엔진을 생성한 뒤 임의의 입력값으로 추론을 실행해 순수 엔진 성능을 측정했다.

| 항목 | 측정 결과 |
|---|---:|
| 처리량 | **235.803 queries/s** |
| 평균 전체 지연 시간 | **4.76311 ms** |
| 평균 GPU 연산 시간 | **4.22742 ms** |
| 중앙값 지연 시간 | 4.61279 ms |
| P95 지연 시간 | 5.49785 ms |
| 생성된 엔진 크기 | 약 8.30 MiB |
| 엔진 빌드 시간 | 약 484.71초 |

이 결과는 YOLO11n 네트워크 자체가 Jetson Orin GPU에서 약 4~5 ms 수준으로 실행될 수 있음을 보여준다.

다만 **235.803 qps가 실제 카메라 화면에서 그대로 235 FPS가 된다는 뜻은 아니다.** 이 측정에는 전체 카메라 파이프라인의 모든 비용이 포함되지 않는다.

실제 End-to-End FPS에는 다음 시간이 추가된다.

- 카메라 프레임 획득
- 이미지 크기 조절 및 정규화
- CPU와 GPU 사이의 데이터 전달
- YOLO 후처리와 NMS
- Bounding Box와 텍스트 그리기
- 화면 출력

따라서 실제 카메라 FPS는 `trtexec` 처리량보다 낮게 측정되는 것이 정상이다.

## 벤치마크 경고의 의미

측정 과정에서 GPU 연산 시간의 변동계수가 **7.92857%**라는 경고가 발생했다. 엔진 생성이나 실행이 실패했다는 의미는 아니며, 반복 추론 시간이 완전히 일정하지 않았다는 뜻이다.

가능한 원인은 다음과 같다.

- GPU 클럭이 실행 중 변함
- 다른 프로세스가 CPU나 GPU를 함께 사용함
- 온도 또는 전력 상태가 변함
- Spin-wait를 사용하지 않아 실행 대기 시간에 편차가 생김

정밀한 비교가 필요하면 동일한 Power Mode와 온도 조건을 유지하고, 측정 시간을 늘리거나 `--useSpinWait` 옵션을 적용하는 것이 좋다.

## 현재 진행 상태

| 단계 | 상태 | 확인 내용 |
|---|:---:|---|
| 객체 탐지 기본 개념 실습 | 완료 | BBox, IoU, NMS 구현 |
| YOLO11n 정적 이미지 추론 | 완료 | 이미지 및 클래스 지정 탐지 |
| 실시간 카메라 코드 준비 | 완료 | GStreamer + OpenCV + YOLO 구조 |
| PyTorch 모델 준비 | 완료 | `yolo11n.pt` 존재 |
| ONNX 변환 | 완료 | `yolo11n.onnx` 존재 |
| FP16 TensorRT 엔진 생성 | 완료 | `yolo11n_fp16.engine` 존재 |
| FP16 기본 벤치마크 | 완료 | `trtexec PASSED` 및 성능 수치 확인 |
| FP16 실시간 카메라 테스트 | 확인 필요 | 실제 FPS와 탐지 품질 측정 필요 |
| `.pt`와 FP16 `.engine` 비교 | 예정 | 동일 조건의 End-to-End FPS 비교 |
| FP32 엔진 비교 | 선택 사항 | 현재 작업 폴더에서 엔진 미확인 |
| INT8 Calibration 및 엔진 생성 | 예정 | 데이터셋과 INT8 엔진 미확인 |

## 바로 다음에 할 일

### 1. FP16 엔진으로 실시간 객체 탐지 실행

Ultralytics에서 FP16 엔진을 직접 불러온다.

```python
from ultralytics import YOLO

model = YOLO("src/models/YOLO/yolo11n_fp16.engine")
```

`.engine`은 이미 NVIDIA GPU용으로 만들어졌기 때문에 `.pt` 모델처럼 `model.to("cuda")`를 호출하지 않는다.

실행하면서 다음 항목을 확인한다.

- 카메라가 정상적으로 열리는가?
- Bounding Box와 클래스가 올바르게 표시되는가?
- 장시간 실행해도 화면이 멈추지 않는가?
- 평균 FPS가 어느 정도인가?
- 객체를 놓치거나 잘못 탐지하는 빈도가 증가하지 않았는가?

### 2. 동일 조건에서 `.pt`와 `.engine` 비교

공정한 비교를 위해 아래 조건을 동일하게 맞춘다.

- 카메라 해상도와 FPS
- YOLO 입력 크기(`imgsz=640`)
- Confidence Threshold(`conf=0.25`)
- IoU Threshold(`iou=0.5`)
- 탐지 클래스 설정
- Jetson Power Mode
- 실행 시간과 장치 온도
- 화면 출력 및 Bounding Box 시각화 여부

권장 기록 양식은 다음과 같다.

| 모델 | 정밀도 | 평균 End-to-End FPS | GPU 사용률 | GPU 온도 | 전력 | 탐지 품질 메모 |
|---|---|---:|---:|---:|---:|---|
| `yolo11n.pt` | FP32 또는 프레임워크 설정 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 기준 모델 |
| `yolo11n_fp16.engine` | FP16 중심 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 최적화 모델 |

### 3. Jetson 자원 상태 확인

실시간 추론과 동시에 다른 터미널에서 다음 명령을 실행한다.

```bash
tegrastats
```

중점적으로 볼 값은 RAM, CPU, `GR3D_FREQ`, CPU/GPU 온도, `VDD_IN`이다. 성능을 비교할 때 Power Mode가 달라지면 결과도 달라지므로 현재 설정도 함께 기록한다.

```bash
sudo nvpmodel -q
```

### 4. 필요할 때 INT8로 확장

FP16 성능이 목표 FPS에 미치지 못하거나 전력과 메모리를 더 줄여야 한다면 INT8 양자화를 진행한다.

INT8은 더 빠를 가능성이 있지만 대표 이미지로 Calibration해야 하며 정확도가 감소할 수 있다. 따라서 먼저 FP16의 실제 성능과 탐지 품질을 확인한 다음 진행하는 것이 적절하다.

## 이번 단계의 완료 기준

다음 항목을 확인하면 FP16 최적화 작업을 완료했다고 볼 수 있다.

- [x] YOLO11n PyTorch 모델 준비
- [x] ONNX 모델 생성
- [x] ONNX 모델을 TensorRT FP16 엔진으로 변환
- [x] `trtexec` 실행 성공 확인
- [x] 순수 엔진 처리량과 지연 시간 기록
- [ ] FP16 엔진으로 CSI 카메라 객체 탐지 실행
- [ ] `.pt` 모델의 End-to-End FPS 기록
- [ ] FP16 `.engine` 모델의 End-to-End FPS 기록
- [ ] 두 모델의 속도와 탐지 품질 비교
- [ ] 실행 중 온도·GPU 사용률·전력 상태 기록

## 핵심 정리

현재 작업의 기술적 핵심은 다음 세 줄로 정리할 수 있다.

1. YOLO11n의 객체 탐지 원리와 실시간 카메라 처리 구조를 학습했다.
2. 원본 PyTorch 모델을 ONNX를 거쳐 Jetson용 TensorRT FP16 엔진으로 성공적으로 변환했다.
3. 엔진 자체 성능은 확인했으며, 이제 실제 카메라 환경에서 `.pt` 대비 FPS 향상과 탐지 품질을 검증해야 한다.

세부 TensorRT 측정값은 [`tensorrt_yolo11n_fp16_benchmark.md`](tensorrt_yolo11n_fp16_benchmark.md)에서 확인할 수 있다.
