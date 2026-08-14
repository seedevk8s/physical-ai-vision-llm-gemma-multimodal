# 05·06 Vision-LLM 실습 실행 및 리뷰 가이드

이 문서는 Jetson Orin이 없는 상황에서도 5번과 6번 실습에서 **무엇을 어떤 순서로 실행했는지**, 각 구성 요소가 **어떻게 연결되는지**, 다시 실행하려면 **무엇을 바꿔야 하는지** 쉽게 검토할 수 있도록 정리한 기록이다.

## 1. 실습 파일과 목표

| 순서 | 파일 | 핵심 목표 |
|---|---|---|
| 5 | `05-LLM_and_Gemma.ipynb` | LLM, 토큰, Context, Prompt, Memory를 이해하고 Gemma를 API 및 로컬 GGUF 모델로 실행 |
| 6-1 | `06-Vision_LLM_Camera_Easy.ipynb` | CSI 카메라 영상에서 YOLO로 객체를 찾고, 탐지 결과를 Gemma가 설명하게 구성 |
| 6-2 | `06-Vision_LLM_Multimodal_Systems.ipynb` | Vision + STT + LLM + TTS를 하나의 음성 질의 시스템으로 통합 |

전체 흐름은 다음과 같다.

```text
5번: 사용자 텍스트 → Gemma → 텍스트 답변

6번 카메라: CSI 카메라 → YOLO → 객체 목록 → Gemma → 텍스트 답변

6번 통합: CSI 카메라 → YOLO ───────────────┐
             마이크 → Whisper → 사용자 질문 ├→ Gemma → Piper → 스피커
                                             ┘
```

> 중요: 이 프로젝트에서 Gemma는 카메라 원본 영상을 직접 보는 것이 아니다. YOLO가 찾은 객체 이름과 신뢰도를 텍스트로 변환한 **Vision Context**를 읽고 답한다. `mmproj`를 사용하는 별도 이미지 기반 예제만 실제 이미지를 멀티모달 모델에 전달한다.

## 2. 공통 실행 환경과 보관 파일

노트북 커널은 프로젝트의 Python 3.10 `.venv`를 사용했다. 노트북은 항상 저장소 최상위 폴더를 작업 경로로 두고 실행해야 상대 경로가 맞는다.

현재 실습에서 참조하는 주요 파일은 다음과 같다.

```text
src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf
src/models/YOLO/yolo11n_int8.engine
src/models/YOLO/yolo11n.pt
src/models/Piper/ko_KR-kss-medium.onnx
src/models/Piper/ko_KR-kss-medium.onnx.json
whisper.cpp/build-cpu/bin/whisper-cli
whisper.cpp/models/ggml-base.bin
src/audio/input.wav
src/audio/response.wav
```

모델 파일은 크기가 크고 Git에 포함되지 않을 수 있다. 나중에 다른 PC에서 리뷰할 때는 노트북과 이 문서를 읽는 것만으로 실행 과정을 확인할 수 있지만, 실제 재실행에는 모델을 별도로 준비해야 한다.

## 3. 5번: LLM과 Gemma

### 3.1 Hugging Face API로 Gemma 호출

초반 실습은 로컬 연산이 아니라 Hugging Face의 원격 추론 서비스를 사용했다.

1. `huggingface_hub`, `python-dotenv`를 설치했다.
2. Hugging Face에서 Gemma 라이선스에 동의하고 Access Token을 발급했다.
3. 저장소 최상위의 `.env`에 다음 형식으로 저장했다.

   ```text
   HF_TOKEN=hf_...
   ```

4. Python에서 `load_dotenv()`와 `os.getenv("HF_TOKEN")`으로 토큰을 읽었다.
5. `InferenceClient`를 이용해 `google/gemma-3-4b-it`에 `messages`를 전달했다.

토큰은 노트북 코드나 문서에 직접 기록하지 않고 `.env`에만 보관해야 한다. `.env`는 `.gitignore`에 포함한다.

### 3.2 Tokenizer와 Context Window 확인

`transformers`의 `AutoTokenizer`로 다음 과정을 확인했다.

- 문자열을 token으로 분리
- token을 정수 token ID로 변환
- `encode()`와 `decode()`로 변환 및 복원
- `<bos>` 같은 special token 확인
- `return_tensors="pt"`로 실제 모델 입력 형태 확인
- 짧은 prompt와 긴 prompt의 token 수 및 API 사용량 비교

핵심은 **입력 token과 생성 token이 모두 Context Window를 사용한다**는 점이다. 입력이 너무 길면 답변을 생성할 여유가 줄어든다.

### 3.3 Prompt와 Memory 실습

Gemma에 전달하는 기본 구조는 다음과 같았다.

```python
messages = [
    {"role": "system", "content": "모델의 역할과 기본 지시"},
    {"role": "user", "content": "사용자의 질문"},
    {"role": "assistant", "content": "이전 모델 답변"},
]
```

Prompt에는 역할, 지시, 질문, 배경 정보, 제한 조건, 출력 형식과 예시를 넣어 결과 변화를 관찰했다. Memory는 모델 내부에 영구 저장되는 기억이 아니라 이전 대화를 다음 요청의 Context에 다시 넣는 방식이다.

- Full History: 전체 대화 유지
- Sliding Window: 최근 몇 개 turn만 유지
- Summary Memory: 오래된 내용을 요약해 유지

### 3.4 로컬 GGUF Gemma 설치와 실행

사용한 모델은 다음 Hugging Face 저장소에서 받았다.

```text
Repository: bartowski/google_gemma-4-E2B-it-GGUF
LLM:        google_gemma-4-E2B-it-Q4_K_M.gguf
Vision:     mmproj-google_gemma-4-E2B-it-f16.gguf
```

`hf_hub_download()`로 두 파일을 `src/models/Gemma4`에 저장했다. Jetson에서 CUDA 가속을 사용하기 위해 `llama-cpp-python`을 CUDA 옵션으로 빌드했다.

```bash
export PATH=/usr/local/cuda/bin:$PATH
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
pip install --no-cache-dir llama-cpp-python
```

5번 노트북의 로컬 모델 설정은 다음과 같았다.

```python
llm = Llama(
    model_path="src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)
```

`create_chat_completion()`에 질문을 전달하고 `response["choices"][0]["message"]["content"]`에서 답변을 꺼냈다. 마지막에는 `exit`, `quit`, `q`, `종료` 중 하나를 입력할 때까지 반복하는 Memory-Less 챗봇을 만들었다. Memory-Less인 이유는 매 질문마다 새 `messages` 목록을 만들어 이전 대화를 전달하지 않았기 때문이다.

> Jetson Linux R36.4.7에서 CUDA 메모리 할당 문제가 발생할 수 있어 당시 노트북에서는 R36.5 이상으로 업데이트하는 절차도 확인했다. 이 절차는 Jetson 전용이며 일반 PC에서는 실행하지 않는다.

## 4. 6번: Vision-LLM 카메라 시스템

### 4.1 객체 탐지 결과를 LLM Context로 변환

YOLO 결과의 `boxes`에서 다음 값을 추출했다.

- `cls`: 객체 class ID
- `conf`: 탐지 신뢰도
- `xyxy`: 바운딩 박스 좌표 `(x1, y1, x2, y2)`
- `names`: class ID에 대응하는 이름

이를 먼저 Python dictionary/JSON 형태로 만들고, 다시 아래와 같은 자연어로 변환했다.

```text
1번 객체는 person이며, confidence는 0.91입니다.
2번 객체는 chair이며, confidence는 0.78입니다.
```

Gemma Prompt에는 이 텍스트를 Context로 넣고 다음 조건을 부여했다.

- 현재 객체 탐지 정보만 근거로 답변
- 탐지되지 않은 객체나 상황을 추측하지 않음
- 한국어 두 문장 이내로 출력

### 4.2 간편 카메라 노트북 실행 순서

`06-Vision_LLM_Camera_Easy.ipynb`는 위에서부터 세 단계로 실행한다.

1. YOLO와 Gemma 모델 로드
2. YOLO 결과 → dictionary → 자연어 변환 함수 정의
3. CSI 카메라 루프 실행

당시 메모리 부족을 줄이기 위해 Gemma는 CPU로 실행했다.

```python
YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=0,
    n_ctx=1024,
    n_batch=8,
    n_ubatch=8,
    verbose=False,
)
```

카메라 설정은 CSI sensor 0, 1280×720, 30 FPS였고 `nvarguscamerasrc` GStreamer pipeline을 사용했다. 카메라 창을 한 번 클릭한 뒤 영문 입력 상태에서 다음 키를 눌렀다.

- `l`: 그 순간의 YOLO 탐지 결과를 Gemma에 질문하고 노트북 출력에서 답변 확인
- `q`: 카메라 종료

LLM이 답변을 만드는 동안 같은 thread에서 처리하므로 카메라 화면이 잠시 멈출 수 있다. 종료 시 `camera.release()`와 `cv2.destroyAllWindows()`가 실행된다.

## 5. 6번: STT와 TTS 준비

### 5.1 Whisper.cpp 음성 인식

`whisper.cpp`를 저장소 안에 clone한 뒤 CPU용 `whisper-cli`를 빌드했다.

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
sudo apt install -y cmake
cmake -B build-cpu
cmake --build build-cpu --target whisper-cli -j4 --config Release
sh ./models/download-ggml-model.sh base
```

5초 녹음과 한국어 STT의 핵심 명령은 다음과 같다.

```bash
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 5 ../src/audio/input.wav

./build-cpu/bin/whisper-cli \
  -m models/ggml-base.bin \
  -f ../src/audio/input.wav \
  -l ko \
  --no-timestamps
```

통합 코드에서는 `pasuspender -- arecord ...`로 PulseAudio가 녹음 장치를 점유하지 않도록 했다.

### 5.2 Piper 한국어 음성 합성

Whisper 폴더 안에 Piper 전용 가상환경을 만들고 음성을 내려받았다.

```bash
cd whisper.cpp
python3 -m venv .piper_venv
source .piper_venv/bin/activate
pip install piper-tts
cd ..
mkdir -p src/models/Piper
python3 -m piper.download_voices \
  ko_KR-kss-medium \
  --data-dir src/models/Piper
```

Piper는 답변을 `src/audio/response.wav`로 만들고 `aplay`가 스피커로 재생했다.

```bash
whisper.cpp/.piper_venv/bin/python -m piper \
  -m src/models/Piper/ko_KR-kss-medium.onnx \
  -f src/audio/response.wav \
  -- "앞에 사람이 한 명 있습니다."

aplay src/audio/response.wav
```

## 6. 최종 Vision + STT + LLM + TTS 실행

최종 셀의 실제 설정은 다음과 같았다.

| 항목 | 값 |
|---|---|
| YOLO | `src/models/YOLO/yolo11n_int8.engine` |
| Gemma | `src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf` |
| Whisper 실행 파일 | `whisper.cpp/build-cpu/bin/whisper-cli` |
| Whisper 모델 | `whisper.cpp/models/ggml-base.bin` |
| Piper Python | `whisper.cpp/.piper_venv/bin/python` |
| Piper 음성 | `src/models/Piper/ko_KR-kss-medium.onnx` |
| 마이크 | `plughw:1,0` |
| 스피커 | `plughw:0,0` |
| 녹음 시간 | 5초 |
| 카메라 | CSI sensor 0, 1280×720, 30 FPS |
| Gemma | CPU, context 1024, batch 8, 최대 출력 80 token |

실행 절차는 다음과 같다.

1. `06-Vision_LLM_Multimodal_Systems.ipynb`의 최종 통합 셀을 `Shift + Enter`로 실행한다.
2. YOLO와 Gemma가 로드되고 별도의 OpenCV 카메라 창이 열린다.
3. 카메라 창을 클릭해 키보드 focus를 준다.
4. 영문 입력 상태에서 `v`를 누른다.
5. 터미널에 `질문을 말씀해 주세요!`가 나오면 5초 안에 질문한다.
6. Whisper가 질문을 텍스트로 변환한다.
7. `v`를 누른 순간의 YOLO 탐지 결과와 질문을 Gemma에 함께 전달한다.
8. Gemma 답변이 노트북에 출력된다.
9. Piper가 답변 WAV를 만들고 스피커로 재생한다.
10. 카메라 창에서 `q`를 누르면 종료한다.

예를 들면 카메라가 사람과 의자를 찾은 상태에서 `v`를 누르고 “지금 앞에 무엇이 있어?”라고 말한다. Whisper가 질문을 인식하고, Gemma는 `person`, `chair`라는 Vision Context 안에서만 한국어 두 문장 이내로 답한다.

## 7. Jetson 없이 나중에 리뷰하거나 재실행하는 방법

### 7.1 코드와 실행 원리만 리뷰

Jetson이 없어도 세 노트북과 이 문서만 있으면 전체 실행 순서를 검토할 수 있다. 특히 다음 함수를 순서대로 읽으면 최종 pipeline이 명확하다.

```text
project_detections_to_text()  YOLO 결과를 텍스트로 변환
project_speech_to_text()      녹음 후 Whisper로 질문 인식
project_ask_llm()             질문 + Vision Context를 Gemma에 전달
project_text_to_speech()      Gemma 답변을 Piper 음성으로 생성/재생
while True                    카메라, v/q 입력과 전체 함수 연결
```

Jetson 전용 기능은 실제로 동작하지 않아도 코드 구조와 저장된 notebook output을 통해 리뷰할 수 있다.

### 7.2 NVIDIA GPU 일반 PC에서 재실행

다음 두 부분을 반드시 바꿔야 한다.

1. TensorRT `.engine`은 생성한 GPU/JetPack/TensorRT 환경에 종속적이므로 기존 Jetson engine을 그대로 사용하지 않는다. `yolo11n.pt`를 사용하거나 PC에서 engine을 다시 export한다.
2. `nvarguscamerasrc`는 Jetson CSI 전용이므로 USB/web camera라면 `cv2.VideoCapture(0)`으로 바꾼다.

예시:

```python
YOLO_MODEL_PATH = "src/models/YOLO/yolo11n.pt"
camera = cv2.VideoCapture(0)
```

Gemma는 PC의 메모리와 GPU 환경에 따라 `n_gpu_layers=-1` 또는 `0`을 선택한다. CUDA용 `llama-cpp-python`은 그 PC의 CUDA 버전에 맞춰 다시 설치한다.

### 7.3 GPU 없는 일반 PC에서 재실행

- YOLO: `.pt` 모델을 CPU에서 실행
- Gemma: `n_gpu_layers=0`으로 GGUF 모델을 CPU에서 실행
- 카메라: USB camera는 `cv2.VideoCapture(0)` 사용
- STT: `whisper.cpp/build-cpu`를 해당 PC에서 다시 빌드
- TTS: Piper 가상환경을 해당 PC에서 다시 생성

동작은 가능하지만 YOLO와 Gemma 응답이 Jetson/GPU보다 느릴 수 있다. RAM은 Gemma GGUF 파일 크기뿐 아니라 context와 내부 buffer까지 감당할 여유가 있어야 한다.

### 7.4 카메라 없이 pipeline 리뷰

저장된 이미지 한 장으로 카메라 부분만 대체할 수 있다.

```python
frame = cv2.imread("test.jpg")
detection = yolo.predict(frame, conf=0.25, iou=0.5, verbose=False)[0]
vision_text = project_detections_to_text(detection)
answer = project_ask_llm("사진에 무엇이 있어?", vision_text)
print(answer)
```

마이크가 없으면 `project_speech_to_text()` 대신 질문 문자열을 직접 넣고, 스피커가 없으면 `project_text_to_speech()`를 생략한 뒤 생성된 텍스트만 확인한다. 이 방식이면 Jetson 장치 없이도 핵심인 `Vision → Context → LLM` 연결을 검증할 수 있다.

## 8. 자주 발생하는 문제와 확인 방법

### 카메라가 열리지 않음

- 다른 notebook cell이나 process가 카메라를 점유하는지 확인한다.
- Jetson에서는 CSI cable, `sensor-id`, `nvargus-daemon` 상태를 확인한다.
- 원격 SSH/Jupyter에서는 GUI display 연결이 없으면 `cv2.imshow()`가 열리지 않는다.
- 일반 PC에서는 Jetson GStreamer pipeline 대신 `cv2.VideoCapture(0)`을 사용한다.

### `v`, `l`, `q` 키가 반응하지 않음

- Jupyter cell이 아니라 별도로 열린 OpenCV 카메라 창을 먼저 클릭한다.
- 한글이 아닌 영문 입력 상태인지 확인한다.
- 간편 notebook은 `l`, 최종 통합 notebook은 `v`, 종료는 둘 다 `q`이다.

### 마이크 또는 스피커 장치 오류

장치 번호는 연결 순서에 따라 달라질 수 있다.

```bash
arecord -l
aplay -l
```

출력에서 실제 card/device 번호를 찾아 `MIC_DEVICE`와 `SPEAKER_DEVICE`를 수정한다. `plughw:1,0`, `plughw:0,0`은 당시 환경의 값이지 모든 컴퓨터의 고정값이 아니다.

### CUDA out-of-memory 또는 시스템 멈춤

- `n_gpu_layers=0`으로 Gemma를 CPU에 배치한다.
- `n_ctx`, `n_batch`, `n_ubatch`, `MAX_TOKENS`를 줄인다.
- YOLO와 Gemma가 Jetson의 통합 메모리를 함께 사용한다는 점을 고려한다.
- 다른 notebook에서 로드한 모델과 카메라를 먼저 종료하고 kernel을 재시작한다.

### TensorRT engine 오류

`.engine`은 범용 모델 파일이 아니다. 다른 장치에서는 원본 `.pt` 또는 `.onnx`를 사용하거나 대상 환경에서 다시 build/export한다.

## 9. 나중을 위한 보존 체크리스트

- [ ] 세 notebook과 이 Markdown 문서를 함께 보관
- [ ] Python 및 JetPack/CUDA/TensorRT 버전을 별도 기록
- [ ] `pip freeze` 결과를 보관
- [ ] 모델의 repository, 정확한 filename과 checksum을 기록
- [ ] 마이크/스피커의 `arecord -l`, `aplay -l` 결과를 기록
- [ ] 카메라 종류, sensor ID, 해상도와 FPS를 기록
- [ ] 정상 실행 화면과 notebook output을 캡처
- [ ] `.env`와 Access Token은 공유 자료에서 제외
- [ ] 대용량 model/engine 파일의 별도 backup 위치를 기록

이 문서에서 가장 중요한 재현 포인트는 **Jetson 전용 TensorRT engine과 CSI camera pipeline은 다른 컴퓨터에서 그대로 재사용할 수 없지만**, `.pt` 모델과 일반 camera 입력으로 바꾸면 나머지 `YOLO → text Context → Gemma → 음성 출력` 구조는 그대로 리뷰하고 재구성할 수 있다는 점이다.
