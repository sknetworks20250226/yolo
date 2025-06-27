mvtec- 이미지 제공


# RoboFlow 사용법

1. 프로젝트 생성
    - 타입 선택: 분류, 포즈 등
2. 이미지 업로드
3. 라벨링
    - 태그 지정 가능
    - 어느 정도 라벨링 하면 오토 라벨링 가능한 것 같음
4. 데이터셋에 추가
    - 훈련용, 검증용, 테스트용 비율에 맞게 추가
5. data set version 생성하여 데이터셋 파일 생성
6. 데이터셋 파일 설치
7. yolo detect train data=My-First-Project-3/data.yaml model=yolo11n.pt epochs=50 imgsz=640 이런 형식으로 터미널에서 명령 실행
8. roboflow에서 프로젝트의 versions에서 훈련 진행 상황 보기 가능


## 학습 실행 코드

- yolo detect train data=My-First-Project-3/data.yaml model=yolo11n.pt epochs=50 imgsz=640

| 항목 | 의미 |
| --- | --- |
| `yolo` | YOLO 커맨드라인 툴 (Ultralytics에서 제공) |
| `detect` | 작업 유형 (객체 탐지, 즉 바운딩 박스 기반 탐지) |
| `train` | 학습 모드 실행 |
| `data=My-First-Project-3/data.yaml` | 데이터셋 구성 정보가 들어있는 YAML 파일 경로 |
| `model=yolo11n.pt` | 사용할 사전학습(pretrained) 모델 → 여기서부터 미세조정(fine-tuning) |
| `epochs=50` | 학습을 50회 반복 |
| `imgsz=640` | 입력 이미지 크기를 640x640으로 리사이즈 후 학습 |


## 예측

```python
# predict.ipynb
from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")
results = model("010.png", conf=0.3)

# 결과 이미지 보기
results[0].show()

# 결과 저장
results[0].save(filename='output.jpg')
```

이런 식으로 훈련한 모델 가져와서 예측 가능


### 터미널 명령

#### 이미지 1 장 예측

- yolo detect predict model=runs/detect/train/weights/best.pt source=sample.jpg

#### 폴더 안 이미지들 일괄 예측

- yolo detect predict model=runs/detect/train/weights/best.pt source=My-First-Project-1/valid/images

| 옵션 | 설명 |
| --- | --- |
| `conf=0.25` | 신뢰도(confidence) 임계값 (기본: 0.25) |
| `save=True` | 결과 이미지 저장 여부 (기본: 저장됨) |
| `show=True` | 결과 이미지 즉시 보기 (Jupyter에서는 표시됨) |
| `project=...` | 결과 저장할 상위 폴더 지정 |
| `name=...` | 결과 저장할 하위 폴더 이름 지정 |