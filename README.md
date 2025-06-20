# yolo
# 제작 구현 스텝
## Step 1 프로젝트 만들기
![](/Yolo1차/산출물/1.png)
roboflow 메인 화면에서 오른쪽 상단에 있는 New Project 버튼을 눌러서 새로운 Project를 생성합니다.
![](/Yolo1차/산출물/2.png)
프로젝트 초기 설정입니다.<br>
Project Name과 그룹이름을 설정합니다. (그룹은 이미지의 라벨 카테고리를 말하면서 이후에 다시 수정, 추가 할 수 있습니다.)
## Step 2 데이터 업로드
![](/Yolo1차/산출물/3.png)
라벨링 할 파일 또는 파일들이 들어있는 폴더를 선택합니다.
![](/Yolo1차/산출물/4.png)
파일들이 로드가 다 됐다면 Save And Continue 버튼을 눌러서 다음으로 넘어갑니다.
## Step 3 데이터 라벨링
![](/Yolo1차/산출물/5.png)
라벨링 시작 설정입니다. 라벨링 빠른 시작과 자동 라벨링 기능이 있습니다.
![](/Yolo1차/산출물/6.png)
작업자를 지목하고 일을 분배하는 설정입니다. 스스로에게 분배를 선택했습니다.
![](/Yolo1차/산출물/7.png)
업로드했던 파일이 보입니다. 더블클릭해서 라벨링하는 곳으로 이동합니다.
![](/Yolo1차/산출물/8.png)
화면에 타원안쪽 영역이 라벨링을 할 수 있도록 돕는 도구들의 종류입니다.<br>
첫번째는 기존에 한 라벨을 이동하는 도구<br>
두번째는 사각형의 라벨링을 하는 도구<br>
세번째는 다각형의 라벨링을 할 수 있게 해주는 도구<br>
그 외에도 roboflow에서 지원하는 자동 라벨링 도구와 사용자가 제작한 모델로 자동 라벨링을 하는 도구가 있습니다.
![](/Yolo1차/산출물/9.png)
사각형으로 라벨링을 했고, 라벨링이 지역을 설정이 끝나면 자동으로 왼쪽 상단에 그룹의 이름을 정할 수 있는 창이 나옵니다.<br>
이름을 설정하고 Save 버튼을 클릭해서 저장합니다.
![](/Yolo1차/산출물/10.png)
다각형 도구로 작업했을때의 예시입니다.
![](/Yolo1차/산출물/11.png)
라벨링 작업이 끝났을 때의 모습입니다.
## Step 4 데이터셋 저장하기
![](/Yolo1차/산출물/12.png)
모든 라벨링이 끝났다면 뒤로 나가면 됩니다. 자동으로 저장이 됩니다.<br>
왼쪽에 있는 메뉴에서 Annotate 창에 있다면 Add nImage to Dataset 버튼을 눌러서 데이터 셋을 저장합니다.
![](/Yolo1차/산출물/13.png)<br>
버튼을 누르면 해당 창을 볼 수 있습니다.<br>
Method에서 Train, Valid, Test 데이터들의 각각의 비율을 정해서 데이터 셋을 만들 수 있습니다.
![](/Yolo1차/산출물/14.png)
이번에는 왼쪽에 있는 메뉴에서 Versions을 선택합니다.<br>
마지막으로 데이터를 전처리 할 수 있는 창입니다. 기호에 맞게 설정하면 됩니다.<br>
해당 예시에서는 따로 다른 설정을 하지 않았습니다.
![](/Yolo1차/산출물/15.png)
설정이 끝나면 해당창이 보이고 여기서 Download Dataset을 선택합니다.
![](/Yolo1차/산출물/16.png)
데이터셋을 저장하기 전에 모델 포멧을 설정해야합니다. 예시에서는 Yolo v8을 선택했습니다.<br>
Download Options는 Zip파일과 노트북파일에서 코드로 다운 받을 수 있는 2개의 선택지가 있습니다.<br>
예시에서는 코드로 받는 선택을 하고 넘어갔습니다.
![](/Yolo1차/산출물/17.png)
보여지는 코드를 노트북에 치고 실행하면 데이터셋을 다운받게 됩니다.
## Step 5 노트북으로 데이터 받기
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="wm5*****************")
project = rf.workspace("reenactpigmailcom").project("test-65cwb")
version = project.version(1)
dataset = version.download("yolov8")
```
위 코드를 실행하면 제작한 데이터셋들을 받을 수 있습니다.
## Step 6 학습하기(학습만 코랩환경)
```
!pip install ultralytics
from ultralytics import YOLO
from google.colab import files

# 사전 학습된 모델 로드
model = YOLO("yolov8n.pt")

# 학습 시작
model.train(
    data="/content/Hot6TheKingRush-factory-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4
)
files.download("/content/runs/detect/train/weights/best.pt")
```
## Step 7 출력, 테스트하기
```
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 추론 결과는 1개니까 바로 r로 꺼내기
    r = model(frame, imgsz=640, stream=False)[0]

    # confidence 0.8 이상 필터링
    for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        if conf >= 0.8:
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 - 0.8 이상만 감지", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

```






