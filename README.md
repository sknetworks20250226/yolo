# roboflow 활용하여 yolo 학습하기


## roboflow 활용

### 1. 계정 생성 및 로그인

![image](https://github.com/user-attachments/assets/b860467b-2b0c-4c0e-ae9e-9aaf9801463f)

### 2. workspace 생성

![2](https://github.com/user-attachments/assets/750fc7ec-f06e-4dae-83ca-470dba18783b)

### 3. project 생성

![3](https://github.com/user-attachments/assets/bbd40f58-84fb-489b-80fb-ebaef204772f)



#### 3-1 각 프로젝트 목적에 맞게 선택

![4](https://github.com/user-attachments/assets/1b42175d-87dc-45d6-9287-4079ad4b393a)


- 객체 탐지
- 분류
- instance 세그멘테이션
- keypoint 탐지
- 멀티 모달

### 4. 파일 또는 폴더 선택하여 사용할 데이터 업로드

![5](https://github.com/user-attachments/assets/14f76ea6-1aff-42d5-9056-720f91b374b5)


![6](https://github.com/user-attachments/assets/e778cc06-e422-4dc4-b97b-dfec89b42ed4)


![7](https://github.com/user-attachments/assets/656c13b2-1c5e-4eae-bbc7-0ea4d04afaf0)


### 5. 데이터 라벨링(annotation)

- Start labeling 클릭
- 사용할 이미지 수 지정 및 라벨링 참여 인원 설정(혼자면 myself)

![8](https://github.com/user-attachments/assets/124b4fbc-0704-4b48-ab38-db89e5d19a09)


![9](https://github.com/user-attachments/assets/6e72b520-cb6a-449f-984a-56efe72b8e1a)

- 원하는 부분을 격자로 지정하여 구분

![10](https://github.com/user-attachments/assets/fdf81846-a513-4a9e-a635-dac19e1cc0c7)


데이터셋을 원하는 내용으로 라벨링

- After

![11](https://github.com/user-attachments/assets/6ee1ff23-6431-4492-90c9-3ab0364b764f)


- 데이터셋이 이미지 추가
- 학습용, 검증, 확인 용도에 맞게 데이터셋 분할 선택

![12](https://github.com/user-attachments/assets/7fabbf88-24ca-43ca-b3fe-b22a22182288)


- 완성
![13](https://github.com/user-attachments/assets/d9abd692-e13d-42ed-b60d-fca85876d15e)


### 4.2 데이터셋 구성

![14](https://github.com/user-attachments/assets/03f08ff9-503b-4f2e-ac88-f12ece1192e3)


- 전처리. 데이터 증강 여부 설정

![15](https://github.com/user-attachments/assets/10daecaa-8d04-4b0b-a05d-d45f3e05fa60)

### 6. 데이터셋 생성

![16](https://github.com/user-attachments/assets/a38c44fa-a7ae-4c81-aa81-8bd09e7ce664)


![17](https://github.com/user-attachments/assets/a0175ab6-fb96-43f9-bf8b-483999981d46)


### 7. 데이터셋 다운로드

![18](https://github.com/user-attachments/assets/2f3c39c3-4105-41a6-87d3-d0021e4f4ddb)


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

![19](https://github.com/user-attachments/assets/83c471c2-4319-4a59-b659-31e663973ace)
