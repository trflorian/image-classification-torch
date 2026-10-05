import cv2
import torch

from model import ImageClassificationCNN

print("Loading model...", flush=True)
model = ImageClassificationCNN.load_from_checkpoint("checkpoints/model.ckpt")
model.eval().cuda()

print("Opening webcam...", flush=True)
cap = cv2.VideoCapture(0)


def convert_cv2_to_torch(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0

    return torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().cuda()


label_mapping = {"usb_a": 0, "usb_c": 1, "usb_micro": 2, "usb_mini": 3}
label_mapping_inv = {v: k for k, v in label_mapping.items()}
display_labels = {
    "usb_a": "USB-A",
    "usb_c": "USB-C",
    "usb_micro": "Micro-USB",
    "usb_mini": "Mini-USB",
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_tensor = convert_cv2_to_torch(frame)

    logits = model(frame_tensor)
    confidences = torch.nn.functional.sigmoid(logits)

    _, predicted_class = torch.max(logits, 1)

    predicted_label = label_mapping_inv[predicted_class.item()]
    predicted_name = display_labels[predicted_label]
    # Keep the readout in its own banner so it never covers the camera image.
    banner_height = 112
    height, width = frame.shape[:2]
    display = cv2.copyMakeBorder(
        frame,
        banner_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )

    cv2.rectangle(display, (8, 8), (width - 9, banner_height - 8), (70, 70, 70), 1)
    cv2.putText(
        display,
        f"Prediction: {predicted_name}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )

    column_width = (width - 32) // len(label_mapping)
    for index, (label, confidence) in enumerate(zip(label_mapping, confidences[0])):
        center_x = 16 + index * column_width + column_width // 2
        color = (80, 255, 80) if label == predicted_label else (220, 220, 220)
        label_size = cv2.getTextSize(display_labels[label], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(
            display,
            display_labels[label],
            (center_x - label_size[0] // 2, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        percent = f"{confidence.item():.0%}"
        percent_size = cv2.getTextSize(percent, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(
            display,
            percent,
            (center_x - percent_size[0] // 2, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("frame", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
