# CIFAR-10 CNN GPU 학습부터 모델 저장까지

이 문서는 `03_DL-and-GPU.ipynb`의 **12) CNN 모델 학습부터 마지막 셀까지** 무엇을 수행하는지 초심자 관점에서 설명합니다.

이 구간의 최종 목표는 다음과 같습니다.

> CIFAR-10 이미지를 분류하도록 CNN을 GPU에서 학습하고, 학습 결과를 평가·시각화한 뒤, 학습한 모델을 파일로 저장하고 다시 불러와 사용하는 것

---

## 1. 전체 과정 한눈에 보기

전체 실행 흐름은 다음과 같습니다.

```text
학습 준비 완료
    ↓
GPU에서 5 Epoch 학습
    ↓
테스트 데이터로 매 Epoch 평가
    ↓
단일 이미지 추론
    ↓
여러 이미지 추론 및 오답 확인
    ↓
Loss와 Accuracy 그래프 확인
    ↓
모델과 학습 정보를 .pth 파일로 저장
    ↓
저장된 모델을 다시 불러와 추론
```

여기서 **학습(training)** 은 모델의 가중치를 수정하는 과정이고, **평가(evaluation)** 또는 **추론(inference)** 은 가중치를 수정하지 않고 예측만 수행하는 과정입니다.

---

## 2. 이 모델이 분류하는 데이터

CIFAR-10은 32×32 크기의 RGB 컬러 이미지 데이터셋입니다. 다음 10개 클래스를 구분합니다.

| 번호 | 클래스 | 의미 |
|---:|---|---|
| 0 | airplane | 비행기 |
| 1 | automobile | 자동차 |
| 2 | bird | 새 |
| 3 | cat | 고양이 |
| 4 | deer | 사슴 |
| 5 | dog | 개 |
| 6 | frog | 개구리 |
| 7 | horse | 말 |
| 8 | ship | 배 |
| 9 | truck | 트럭 |

모델에는 이미지가 `[Batch, 3, 32, 32]` 형태로 들어갑니다.

- `Batch`: 한 번에 처리하는 이미지 수
- `3`: RGB 색상 채널
- `32, 32`: 이미지의 높이와 너비

모델의 최종 출력은 `[Batch, 10]` 형태입니다. 이미지마다 10개 클래스에 대한 점수인 **Logit**을 출력한다는 뜻입니다.

---

## 3. 학습에 사용되는 핵심 구성 요소

12번 셀 이전에 다음 객체가 준비되어 있어야 합니다.

### `model`

`CIFAR10CNN`으로 만든 CNN 모델입니다. 다음과 같은 일을 합니다.

1. Convolution으로 이미지 특징을 추출합니다.
2. Batch Normalization으로 학습을 안정화합니다.
3. ReLU로 비선형성을 추가합니다.
4. Max Pooling으로 공간 크기를 줄입니다.
5. Global Average Pooling으로 채널별 특징을 하나의 값으로 요약합니다.
6. Linear Layer가 10개 클래스의 Logit을 출력합니다.

모델은 다음 코드에 의해 GPU 메모리로 이동한 상태입니다.

```python
device = torch.device("cuda")
model = CIFAR10CNN().to(device)
```

### `train_loader`와 `test_loader`

DataLoader는 전체 데이터셋을 작은 Batch로 나누어 모델에 공급합니다.

- `train_loader`: 가중치를 학습할 데이터 제공
- `test_loader`: 학습하지 않은 데이터로 일반화 성능 평가

### `criterion`

```python
criterion = nn.CrossEntropyLoss()
```

모델의 예측과 정답 사이의 차이를 **Loss**라는 숫자로 계산합니다. Loss가 작을수록 일반적으로 예측이 정답에 가까워진 것입니다.

`CrossEntropyLoss`는 내부에서 Logit을 적절히 변환하므로, 학습할 때 모델 출력에 `softmax()`를 먼저 적용하지 않습니다.

### `optimizer`

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
)
```

Optimizer는 역전파로 계산한 Gradient를 이용해 모델의 가중치를 갱신합니다. `lr`은 학습률로, 한 번에 가중치를 얼마나 수정할지 결정합니다.

---

## 4. 12) CNN 모델 학습

### Epoch란?

```python
EPOCHS = 5
```

**1 Epoch**는 모델이 전체 학습 데이터를 한 번 모두 확인했다는 뜻입니다. 따라서 5 Epoch 학습은 전체 학습 데이터셋을 다섯 번 반복해서 학습하는 것입니다.

### 학습 기록 리스트

```python
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []
```

각 Epoch가 끝날 때 Loss와 Accuracy를 저장합니다. 이 값은 나중에 학습 추세를 그래프로 그리는 데 사용합니다.

### GPU 동기화와 시간 측정

```python
torch.cuda.synchronize()
training_start_time = time.perf_counter()
```

GPU 연산은 기본적으로 CPU와 비동기 방식으로 실행됩니다. 즉, CPU가 GPU 작업의 완료를 기다리지 않고 다음 코드로 넘어갈 수 있습니다.

따라서 정확한 시간을 측정하려면 `torch.cuda.synchronize()`로 이전 GPU 작업이 끝날 때까지 기다려야 합니다.

### 한 Epoch의 학습

```python
train_loss, train_accuracy = train_one_epoch(
    model=model,
    data_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
)
```

`train_one_epoch()` 내부에서는 각 Batch마다 다음 작업을 수행합니다.

```text
이미지와 정답을 GPU로 이동
    ↓
이전 Gradient 초기화
    ↓
순전파로 Logit 계산
    ↓
정답과 비교하여 Loss 계산
    ↓
역전파로 Gradient 계산
    ↓
Optimizer가 가중치 갱신
```

중요한 코드는 다음 세 줄입니다.

```python
optimizer.zero_grad()  # 이전 Batch의 Gradient 제거
loss.backward()        # 역전파로 현재 Gradient 계산
optimizer.step()       # 계산된 Gradient로 가중치 갱신
```

### 한 Epoch의 평가

```python
test_loss, test_accuracy = evaluate(
    model=model,
    data_loader=test_loader,
    criterion=criterion,
    device=device,
)
```

평가에서는 모델의 가중치를 수정하지 않습니다.

```python
model.eval()

with torch.no_grad():
    ...
```

- `model.eval()`: BatchNorm과 Dropout 등을 평가 방식으로 전환
- `torch.no_grad()`: Gradient 계산을 중단하여 메모리 사용량과 연산량 절약

### Epoch 결과 출력

각 Epoch가 끝나면 다음 항목이 출력됩니다.

```text
Epoch 1/5
Train loss
Train accuracy
Test loss
Test accuracy
Time
```

각 지표의 의미는 다음과 같습니다.

| 지표 | 의미 | 일반적으로 기대하는 변화 |
|---|---|---|
| Train loss | 학습 데이터에 대한 오차 | 감소 |
| Train accuracy | 학습 데이터 정확도 | 증가 |
| Test loss | 테스트 데이터에 대한 오차 | 감소 |
| Test accuracy | 테스트 데이터 정확도 | 증가 |
| Time | 해당 Epoch의 학습·평가 시간 | 환경에 따라 다름 |

### 전체 학습 시간

```python
torch.cuda.synchronize()
total_training_time = time.perf_counter() - training_start_time
```

마지막 GPU 작업까지 끝난 후, 5 Epoch의 학습과 평가에 걸린 전체 시간을 계산합니다.

---

## 5. 학습된 모델로 이미지 한 장 예측하기

```python
image, true_label = test_dataset[0]
input_batch = image.unsqueeze(0).to(device)
```

`test_dataset[0]`에서 얻은 이미지의 형태는 `[3, 32, 32]`입니다. 그러나 모델은 Batch 차원을 포함한 4차원 입력을 기대하므로 `unsqueeze(0)`을 사용합니다.

```text
[3, 32, 32]
      ↓ unsqueeze(0)
[1, 3, 32, 32]
```

여기서 첫 번째 차원의 `1`은 이미지 한 장으로 이루어진 Batch라는 뜻입니다.

### Logit을 확률로 변환

```python
logits = model(input_batch)
probabilities = torch.softmax(logits, dim=1)
predicted_label = probabilities.argmax(dim=1).item()
```

- `logits`: 모델이 출력한 클래스별 원시 점수
- `softmax`: 10개 점수를 합이 1인 확률로 변환
- `argmax`: 가장 높은 확률을 가진 클래스 번호 선택
- `item()`: Tensor 안의 값 하나를 Python 숫자로 변환

학습 단계와 달리, 추론에서는 사람이 이해하기 쉬운 신뢰도를 표시하기 위해 `softmax()`를 사용합니다.

### Confidence 해석 시 주의점

Confidence가 90%라고 해서 실제로 항상 90%의 확률로 옳다는 뜻은 아닙니다. 이는 모델이 10개 후보 중 해당 클래스에 부여한 상대적인 확신입니다. 모델은 틀린 답을 높은 Confidence로 예측할 수도 있습니다.

### 이미지 역정규화

학습 전에 이미지를 평균과 표준편차로 정규화했으므로, 화면에 자연스러운 색으로 출력하려면 원래 범위로 되돌려야 합니다.

```python
image_denormalized = image.squeeze(0) * std + mean
image_rgb = image_denormalized.permute(1, 2, 0).clamp(0, 1)
```

핵심 변환은 다음과 같습니다.

```text
정규화:   (원본 - mean) / std
역정규화: 정규화된 값 × std + mean
```

`permute(1, 2, 0)`은 PyTorch의 `[Channel, Height, Width]`를 Matplotlib이 사용하는 `[Height, Width, Channel]`로 바꿉니다.

---

## 6. 여러 이미지의 예측 결과 확인하기

테스트 이미지 40장을 순서대로 예측하고 4×10 격자로 표시합니다.

```python
title_color = "red" if predicted_label != true_label else "black"
```

- 검은 제목: 정답을 맞힌 이미지
- 빨간 제목: 잘못 분류한 이미지

각 이미지 위에는 다음 정보가 표시됩니다.

- `True`: 실제 정답
- `Pred`: 모델이 예측한 클래스
- `Conf`: 예측 클래스에 부여한 Confidence

오답 이미지를 관찰하면 모델이 어떤 클래스를 자주 혼동하는지 알 수 있습니다. 예를 들어 고양이와 개, 자동차와 트럭처럼 형태가 비슷한 클래스는 비교적 혼동하기 쉽습니다.

현재 설정은 5 Epoch만 학습하므로 충분히 수렴하지 않았을 수 있습니다. 오답이 존재한다고 해서 코드가 잘못된 것은 아닙니다.

---

## 7. Loss 그래프 해석하기

```python
plt.plot(epoch_axis, train_loss_history, label="Train loss")
plt.plot(epoch_axis, test_loss_history, label="Test loss")
```

Loss 그래프에서는 다음 패턴을 확인합니다.

### 정상적으로 학습되는 경우

- Train loss가 점차 감소
- Test loss도 함께 감소

### 과적합이 의심되는 경우

- Train loss는 계속 감소
- Test loss는 어느 시점부터 증가

과적합은 모델이 학습 데이터를 지나치게 외워 새로운 테스트 이미지에는 잘 대응하지 못하는 현상입니다.

### 학습이 잘 진행되지 않는 경우

- Train loss가 거의 감소하지 않음
- Accuracy도 낮은 수준에 머무름

이 경우 학습률, 모델 구조, 정규화, Epoch 수 또는 데이터 처리 과정을 점검할 수 있습니다.

---

## 8. Accuracy 그래프 해석하기

```python
np.array(train_accuracy_history) * 100
np.array(test_accuracy_history) * 100
```

Accuracy는 0~1 값으로 저장되어 있으므로 그래프에서는 100을 곱해 백분율로 표시합니다.

### 이상적인 흐름

- Train accuracy가 점차 증가
- Test accuracy도 함께 증가
- 두 정확도의 차이가 지나치게 크지 않음

### 과적합이 의심되는 흐름

- Train accuracy는 매우 높음
- Test accuracy는 낮거나 더 이상 증가하지 않음

Train과 Test 결과가 완전히 같을 필요는 없습니다. 학습 데이터로 직접 가중치를 수정했으므로 Train accuracy가 조금 더 높은 것은 자연스럽습니다.

또한 5개의 점만으로 장기적인 학습 추세를 확정하기는 어렵습니다. 더 정확한 판단이 필요하다면 Epoch 수를 늘리고 변화를 관찰해야 합니다.

---

## 9. 학습된 모델 저장하기

학습이 끝나면 다음 정보를 Dictionary에 모아 저장합니다.

```python
save_data = {
    "model_name": "CIFAR10CNN",
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "test_accuracy": test_accuracy_history[-1],
    "epochs": EPOCHS,
    "class_names": CIFAR10_CLASSES,
    "mean": CIFAR10_MEAN,
    "std": CIFAR10_STD,
}
```

| 저장 항목 | 역할 |
|---|---|
| `model_name` | 사용한 모델 이름 |
| `model_state_dict` | 학습된 Weight와 Bias |
| `optimizer_state_dict` | Optimizer의 내부 상태 |
| `test_accuracy` | 마지막 Epoch의 테스트 정확도 |
| `epochs` | 완료한 Epoch 수 |
| `class_names` | 클래스 번호와 이름의 대응 관계 |
| `mean`, `std` | 입력 이미지 정규화 정보 |

그 후 `torch.save()`로 파일을 생성합니다.

```python
torch.save(save_data, "src/models/CIFAR10/CIFAR10_CNN.pth")
```

`.pth` 파일에는 모델 클래스의 Python 코드 자체가 아니라, 주로 모델의 학습된 상태와 부가 정보가 저장됩니다.

저장하기 전에 대상 폴더가 존재해야 합니다. 폴더가 없다면 먼저 다음 코드를 실행할 수 있습니다.

```python
from pathlib import Path

Path("src/models/CIFAR10").mkdir(parents=True, exist_ok=True)
```

---

## 10. 저장된 모델 불러오기

먼저 저장할 때와 동일한 구조의 빈 모델을 만듭니다.

```python
loaded_model = CIFAR10CNN()
```

그 후 체크포인트를 CPU 메모리로 읽습니다.

```python
checkpoint = torch.load(
    "src/models/CIFAR10/CIFAR10_CNN.pth",
    map_location="cpu",
    weights_only=True,
)
```

노트북에서는 이 Dictionary 변수의 이름을 `state_dict`로 사용하지만, 실제 내용은 여러 정보를 포함한 **체크포인트 전체**이므로 `checkpoint`라는 이름이 더 이해하기 쉽습니다.

학습된 파라미터를 새 모델에 적용합니다.

```python
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.to(device)
loaded_model.eval()
```

각 줄의 역할은 다음과 같습니다.

1. `load_state_dict()`: 저장된 Weight와 Bias를 빈 모델에 적용
2. `to(device)`: 모델을 추론에 사용할 GPU로 이동
3. `eval()`: 모델을 평가 모드로 전환

이후 `loaded_model`로 다른 테스트 이미지 40장을 추론합니다. 저장 전 모델과 같은 방식으로 동작한다면 체크포인트가 정상적으로 저장되고 복원된 것입니다.

---

## 11. 학습과 추론의 차이

| 구분 | 학습 | 평가·추론 |
|---|---|---|
| 모드 | `model.train()` | `model.eval()` |
| Gradient 계산 | 사용 | `torch.no_grad()`로 중단 |
| 역전파 | 수행 | 수행하지 않음 |
| 가중치 갱신 | 수행 | 수행하지 않음 |
| Dropout | 일부 값을 무작위로 제거 | 제거하지 않음 |
| BatchNorm | 현재 Batch 통계 사용·갱신 | 저장된 통계 사용 |

`model.eval()`과 `torch.no_grad()`는 역할이 다르므로 추론할 때 둘 다 사용하는 것이 좋습니다.

---

## 12. 자주 묻는 질문

### GPU인데 첫 Epoch가 더 느린 이유는 무엇인가요?

CUDA Context 생성, cuDNN 초기화, 메모리 할당 등이 처음 실행될 때 발생할 수 있습니다. 노트북에서는 실제 측정 전에 GPU Warm-up을 수행해 이 영향을 줄입니다.

### `torch.cuda.synchronize()`는 왜 필요한가요?

GPU는 연산을 비동기로 처리합니다. 동기화 없이 CPU 시간만 측정하면 GPU 작업이 끝나기 전에 측정이 종료되어 실제보다 짧은 시간이 나올 수 있습니다.

### 학습 정확도보다 테스트 정확도가 잠시 높을 수도 있나요?

가능합니다. 학습 모드에서는 Dropout과 데이터 증강 등이 적용될 수 있지만 평가 모드에서는 Dropout이 꺼집니다. 작은 차이는 자연스러울 수 있습니다.

### Loss가 낮으면 Accuracy는 반드시 높나요?

대체로 함께 개선되지만 항상 정확히 같은 방향으로 움직이지는 않습니다. Loss는 정답에 부여한 점수의 크기까지 반영하고, Accuracy는 최종 정답 여부만 계산하기 때문입니다.

### 저장한 파일만 있으면 어디서나 바로 사용할 수 있나요?

동일한 `CIFAR10CNN` 클래스 정의와 호환되는 PyTorch 환경이 필요합니다. 또한 추론 입력에는 학습 때와 같은 정규화 값을 사용해야 합니다.

### 더 좋은 정확도를 얻으려면 어떻게 해야 하나요?

다음 방법을 실험할 수 있습니다.

- Epoch 수 늘리기
- Learning rate 조정하기
- Learning-rate scheduler 사용하기
- Random crop, horizontal flip 같은 데이터 증강 사용하기
- 모델 크기나 구조 조정하기
- Validation set과 Early stopping 사용하기

단, 테스트 데이터를 보며 설정을 반복 조정하면 테스트 데이터에도 간접적으로 과적합될 수 있습니다. 실제 프로젝트에서는 학습·검증·테스트 데이터를 분리하는 것이 좋습니다.

---

## 13. 실행 후 확인 체크리스트

- [ ] `device`가 `cuda`로 출력되는가?
- [ ] Epoch마다 Train/Test loss와 accuracy가 출력되는가?
- [ ] Train loss가 전반적으로 감소하는가?
- [ ] Train/Test accuracy가 전반적으로 증가하는가?
- [ ] 단일 이미지에서 True, Prediction, Confidence가 출력되는가?
- [ ] 여러 이미지 결과에서 오답이 빨간 제목으로 표시되는가?
- [ ] Loss와 Accuracy 그래프가 정상적으로 나타나는가?
- [ ] `src/models/CIFAR10/CIFAR10_CNN.pth`가 생성되는가?
- [ ] 저장된 모델을 불러온 뒤에도 추론이 수행되는가?

---

## 14. 핵심 요약

이 실습은 CNN 정의에서 끝나는 것이 아니라, 실제 딥러닝 프로젝트의 기본 생명주기를 경험하는 과정입니다.

1. CIFAR-10 데이터를 Batch 단위로 GPU에 전달합니다.
2. 순전파, Loss 계산, 역전파, 가중치 갱신을 반복합니다.
3. 매 Epoch마다 테스트 데이터로 일반화 성능을 확인합니다.
4. 학습된 모델로 새로운 이미지를 분류합니다.
5. Loss와 Accuracy 그래프로 학습 상태를 해석합니다.
6. 모델과 관련 정보를 체크포인트로 저장합니다.
7. 체크포인트를 새 모델에 적용해 학습 결과를 재사용합니다.

가장 중요한 점은 **학습 결과 숫자 하나만 보는 것이 아니라**, Train/Test 지표의 흐름, 오답 이미지, Confidence, 저장 후 복원 결과를 함께 확인하는 것입니다.
