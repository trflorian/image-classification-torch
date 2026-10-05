import cv2
import torch

from model import ImageClassificationCNN

model = ImageClassificationCNN.load_from_checkpoint("checkpoints/model.ckpt")
model.eval().cuda()

cap = cv2.VideoCapture(0)


def convert_cv2_to_torch(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0

    return torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().cuda()


label_mapping = {"usb_a": 0, "usb_c": 1, "usb_micro": 2, "usb_mini": 3}
label_mapping_inv = {v: k for k, v in label_mapping.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_tensor = convert_cv2_to_torch(frame)

    logits = model(frame_tensor)
    confidences = torch.nn.functional.sigmoid(logits)

    _, predicted_class = torch.max(logits, 1)

    predicted_label = label_mapping_inv[predicted_class.item()]

    cv2.putText(
        frame,
        ", ".join([f"{label}: {conf:.2f}" for label, conf in zip(label_mapping.keys(), confidences[0])]),
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        predicted_label,
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
