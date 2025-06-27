import cv2
from ultralytics import YOLO

# 1. Colab에서 학습한 모델 불러오기
model = YOLO("best.pt")  # best.pt 파일은 같은 폴더에 있어야 함

# 2. 웹캠 켜기 (0: 기본 노트북 웹캠)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. YOLOv8으로 실시간 추론
    results = model(frame, imgsz=640, stream=True)

    # 4. 결과 시각화
    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLOv8 Webcam Detection", annotated)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()