import cv2
from ultralytics import YOLO


def initialize_model(weights_path="yolov8n.pt"):
    return YOLO(weights_path)


def open_camera(source=1):
    cam = cv2.VideoCapture(source)
    if not cam.isOpened():
        raise RuntimeError("Unable to access camera")
    return cam


def process_frame(model, frame):
    detections = model(frame)
    for detection in detections:
        frame = detection.plot()
    return frame


def main():
    detector = initialize_model()
    camera = open_camera()

    window_name = "Object Detection Stream"

    try:
        while True:
            success, image = camera.read()
            if not success:
                break

            output_frame = process_frame(detector, image)
            cv2.imshow(window_name, output_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
