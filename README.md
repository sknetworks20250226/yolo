# roboflow 활용하여 yolo 학습하기


## roboflow 활용

### 1. 계정 생성 및 로그인

![image.png](attachment:5ca86cf0-9cd4-4b38-a3a1-d761d53874da:image.png)

### ### 2. workspace 생성

![image.png](attachment:a524a09d-0be7-46ee-be29-48ef3ce32f68:image.png)

### 3. project 생성

![image.png](attachment:40c7c4fa-d285-4783-9222-9b282d9b19e8:image.png)

#### 3-1 각 프로젝트 목적에 맞게 선택

![image.png](attachment:1ded30c8-f0f4-47e4-9f95-04d42ef81750:image.png)

- 객체 탐지
- 분류
- instance 세그멘테이션
- keypoint 탐지
- 멀티 모달

### 4. 파일 또는 폴더 선택하여 사용할 데이터 업로드

![image.png](attachment:65300acb-f007-4ad5-98c5-377fe671da94:image.png)

![image.png](attachment:a0f80f21-3af0-4012-97d4-631b1c45c47b:image.png)

![image.png](attachment:6fcfdd8b-7e5b-46a9-bc04-46f1ee1d3a4f:image.png)

### 5. 데이터 라벨링(annotation)

- Start labeling 클릭
- 사용할 이미지 수 지정 및 라벨링 참여 인원 설정(혼자면 myself)

![image.png](attachment:65b5f1f0-0cce-4152-b62d-a8cbdbc1a7e5:image.png)

![image.png](attachment:83bc443d-a43b-4947-a3fe-a1e177ed4f5c:b943a38a-baf5-4627-b629-1626ea64fa8f.png)

- 원하는 부분을 격자로 지정하여 구분

![image.png](attachment:dbd19051-ea53-4ac6-9490-36672267a8a4:image.png)

데이터셋을 원하는 내용으로 라벨링

- After

![image.png](attachment:40901bcf-8a17-4f05-b001-a2a43a9a166d:image.png)

- 데이터셋이 이미지 추가
- 학습용, 검증, 확인 용도에 맞게 데이터셋 분할 선택

![image.png](attachment:e8a58b6d-b422-4d5a-8675-dad23d50758f:image.png)

- 완성

![image.png](attachment:59b5a675-dd41-419c-9386-92f6ada95418:image.png)

### 4.2 데이터셋 구성

![image.png](attachment:b21f7192-9750-4e7a-9e45-6f7ccb1188de:image.png)

- 전처리. 데이터 증강 여부 설정

![image.png](attachment:8a0d8f35-35a4-44e9-93c8-650fde515e18:image.png)

### 6. 데이터셋 생성

![image.png](attachment:91ec2d0b-579c-41c2-b26b-b2fa638286f6:image.png)

![image.png](attachment:811a6fb4-fbc0-4bed-92f2-c03f2f4538fb:image.png)

### 7. 데이터셋 다운로드

![image.png](attachment:9e6f87d7-2db6-42ad-969b-1cad4c951c4c:image.png)

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

![123123.jpg](attachment:dae0c398-c9b3-42e6-a762-486051858ccd:123123.jpg)