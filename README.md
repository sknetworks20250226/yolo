# roboflow 활용하여 yolo 학습하기


## roboflow 활용

### 1. 계정 생성 및 로그인

![alt text](image.png)

### 2. workspace 생성

![alt text](2.png)

### 3. project 생성

![alt text](3.png)


#### 3-1 각 프로젝트 목적에 맞게 선택

![alt text](4.png)


- 객체 탐지
- 분류
- instance 세그멘테이션
- keypoint 탐지
- 멀티 모달

### 4. 파일 또는 폴더 선택하여 사용할 데이터 업로드

![alt text](5.png)

![alt text](6.png)

![alt text](7.png)

### 5. 데이터 라벨링(annotation)

- Start labeling 클릭
- 사용할 이미지 수 지정 및 라벨링 참여 인원 설정(혼자면 myself)

![alt text](8.png)

![alt text](9.png)

- 원하는 부분을 격자로 지정하여 구분

![alt text](10.png)

데이터셋을 원하는 내용으로 라벨링

- After

![alt text](11.png)

- 데이터셋이 이미지 추가
- 학습용, 검증, 확인 용도에 맞게 데이터셋 분할 선택

![alt text](12.png)

- 완성
![alt text](13.png)

### 4.2 데이터셋 구성

![alt text](14.png)

- 전처리. 데이터 증강 여부 설정

![alt text](15.png)

### 6. 데이터셋 생성

![alt text](16.png)

![alt text](17.png)

### 7. 데이터셋 다운로드

![alt text](18.png)

- 데이터셋 다운로드 코드
- 모델 버전에 맞는 다운로드 형식 맞출개걷

```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="1oT8jkxHSKNvu2i8ai1i")
project = rf.workspace("0616-mqqrd").project("0620-fyibr")
version = project.version(2)
dataset = version.download("yolov8")
```

### 8. 데이터셋 학습

```

!pip install ultralytics
from ultralytics import YOLO
from google.colab import files

# 사전 학습된 모델 로드
model = YOLO("yolov8n.pt")

# 학습 시작
model.train(
    data="/content/Hot6TheKingRush-factory-1/data.yaml",  epochs=50,imgsz=640, batch=4)
files.download("/content/runs/detect/train/weights/best.pt")
```

### 9. 모델 출력 테스트하기

```
results = model('/content/test1.jpg', save=True)

```

### 10. 결과

![alt text](19.jpg)