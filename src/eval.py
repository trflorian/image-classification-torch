import cv2
import torch

from model import ImageClassificationCNN

print("Loading model...", flush=True)
model = ImageClassificationCNN.load_from_checkpoint("checkpoints/model.ckpt")
model.eval().cuda()
gradcam_enabled = False
activations = {}
target_layer = model.model.features[-1]
target_layer.register_forward_hook(
    lambda _module, _inputs, output: activations.update(feature_map=output)
)

print("Opening webcam...", flush=True)
cap = cv2.VideoCapture(0)


def convert_cv2_to_torch(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frame = (frame - mean) / std

    return frame.cuda()


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

    activations.clear()
    if gradcam_enabled:
        with torch.enable_grad():
            frame_tensor.requires_grad_(True)
            logits = model(frame_tensor)
            predicted_for_cam = logits.argmax(dim=1).item()
            gradients = torch.autograd.grad(
                logits[0, predicted_for_cam], activations["feature_map"]
            )[0]

        feature_map = activations["feature_map"].detach()[0]
        weights = gradients.detach()[0].mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * feature_map).sum(dim=0))
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        cam = cv2.resize(cam.cpu().numpy(), (frame.shape[1], frame.shape[0]))
        heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    else:
        with torch.no_grad():
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
        f"Prediction: {predicted_name}   Grad-CAM: {'ON' if gradcam_enabled else 'OFF'} (G)",
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

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("g"):
        gradcam_enabled = not gradcam_enabled

cap.release()
cv2.destroyAllWindows()
