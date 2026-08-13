# YOLO11n TensorRT FP16 엔진 빌드 및 벤치마크 결과

## 개요

YOLO11n ONNX 모델을 NVIDIA TensorRT의 `trtexec`로 변환하여 FP16 엔진을 생성하고 추론 성능을 측정했다.

- 실행 일시: 2026-08-13 17:22–17:30 (KST)
- 실행 결과: **성공 (`PASSED`)**
- TensorRT 버전: 10.3.0 (`v100300`)
- 대상 장치: NVIDIA Orin (Compute Capability 8.7)
- 입력 크기: `1 × 3 × 640 × 640`
- 출력 크기: `1 × 84 × 8400`

## 실행 명령어

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx="$HOME/vision-llm/src/models/YOLO/yolo11n.onnx" \
    --saveEngine="$HOME/vision-llm/src/models/YOLO/yolo11n_fp16.engine" \
    --fp16
```

## 입출력 파일

| 구분 | 경로 |
|---|---|
| 입력 ONNX 모델 | `/home/chjin/vision-llm/src/models/YOLO/yolo11n.onnx` |
| 출력 TensorRT 엔진 | `/home/chjin/vision-llm/src/models/YOLO/yolo11n_fp16.engine` |

## 모델 정보

| 항목 | 값 |
|---|---:|
| ONNX IR 버전 | 0.0.9 |
| ONNX Opset | 20 |
| Producer | PyTorch 2.8.0 |
| 네트워크 입력 수 | 1 |
| 네트워크 출력 텐서 수 | 3 |
| 엔진 정밀도 | FP32 + FP16 |
| 생성된 엔진 크기 | 8.29533 MiB |

> `--fp16` 옵션은 FP16 사용을 허용한다. 로그의 `Precision: FP32+FP16`은 TensorRT가 지원 여부와 최적화 결과에 따라 FP16과 FP32 레이어를 함께 사용할 수 있음을 뜻한다.

## 장치 정보

| 항목 | 값 |
|---|---:|
| GPU | NVIDIA Orin |
| SM 수 | 8 |
| GPU 메모리 | 7,619 MiB |
| Shared Memory per SM | 164 KiB |
| Memory Bus Width | 128 bits (ECC disabled) |
| 표시된 Compute Clock | 1.02 GHz |
| 표시된 Memory Clock | 1.02 GHz |

## 엔진 빌드 결과

| 항목 | 결과 |
|---|---:|
| ONNX 파싱 시간 | 0.0366208초 |
| 엔진 생성 시간 | 484.538초 |
| 전체 엔진 빌드 시간 | 484.71초 |
| 역직렬화 시간 | 0.0701625초 |
| 빌드 중 CPU 최대 메모리 | 1,871 MiB |
| TRT allocator GPU 최대 메모리 | 168 MiB |
| 실행 컨텍스트 GPU 메모리 | 9.47266 MiB |
| Activation Memory | 9,932,800 bytes |
| Weights Memory | 5,367,204 bytes |

## 벤치마크 조건

- 입력 데이터: `images` 입력에 임의 값 사용
- 워밍업: 200 ms, 33 queries
- 측정 시간: 약 3.015초
- 측정 queries: 711
- Inference Streams: 1
- 데이터 전송: 활성화
- CUDA Graph: 비활성화
- Spin-wait: 비활성화
- 평균 단위: 10회 추론

## 성능 요약

### 처리량

| 지표 | 결과 |
|---|---:|
| Throughput | **235.803 qps** |
| 총 Host 측정 시간 | 3.01523초 |
| 총 GPU 연산 시간 | 3.0057초 |

### 지연 시간

| 지표 | 최소 | 평균 | 중앙값 | P90 | P95 | P99 | 최대 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전체 Latency (ms) | 4.52142 | **4.76311** | 4.61279 | 5.48114 | 5.49785 | 5.82727 | 7.76685 |
| Enqueue Time (ms) | 2.83618 | 3.08305 | 3.03198 | 3.32001 | 3.49411 | 3.56189 | 3.86028 |
| H2D Latency (ms) | 0.190186 | 0.299738 | 0.282349 | 0.411438 | 0.418518 | 0.42395 | 0.445129 |
| GPU Compute Time (ms) | 4.07581 | **4.22742** | 4.10999 | 4.73355 | 4.74719 | 5.41638 | 7.26752 |
| D2H Latency (ms) | 0.111084 | 0.235956 | 0.220947 | 0.331726 | 0.333374 | 0.340881 | 0.359131 |

## 경고 및 참고 사항

TensorRT는 GPU 연산 시간의 변동계수가 **7.92857%**로 불안정하다는 경고를 출력했다.

```text
GPU compute time is unstable, with coefficient of variance = 7.92857%.
If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
```

보다 안정적인 성능 측정을 위해 다음을 고려할 수 있다.

1. GPU 클럭을 고정한 뒤 다시 측정한다.
2. `--useSpinWait` 옵션을 추가한다.
3. 시스템 부하와 백그라운드 프로세스를 최소화한다.
4. 측정 시간을 늘려 표본 수를 확보한다.

예시:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx="$HOME/vision-llm/src/models/YOLO/yolo11n.onnx" \
    --saveEngine="$HOME/vision-llm/src/models/YOLO/yolo11n_fp16.engine" \
    --fp16 \
    --useSpinWait \
    --duration=10
```

## 결론

YOLO11n ONNX 모델의 TensorRT FP16 엔진 생성과 추론 테스트가 정상적으로 완료되었다. 생성된 엔진은 약 **8.30 MiB**이며, NVIDIA Orin에서 임의 입력 기준 **235.803 qps**, 평균 GPU 연산 시간 **4.22742 ms**, 평균 전체 지연 시간 **4.76311 ms**를 기록했다. 다만 GPU 연산 시간에 일부 편차가 있으므로 정밀 비교용 벤치마크에서는 GPU 클럭 고정 또는 `--useSpinWait` 적용 후 재측정하는 것이 적절하다.
