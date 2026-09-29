"""Практическое занятие 2: детекция и семантическая сегментация кадров.

Запуск из папки cv_lab2: python process.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


ROOT = Path(__file__).resolve().parent
FRAMES = sorted((ROOT / "frames").glob("frame_*.jpg"))
DET_DIR = ROOT / "detection_frames"
SEG_DIR = ROOT / "segmentation_frames"
DATA_DIR = ROOT / "inference_data"
CONFIDENCE = 0.55
COLORS = [(0, 0, 0), (255, 80, 80), (80, 200, 255), (255, 190, 70),
          (180, 110, 255), (80, 220, 130), (255, 100, 210), (220, 220, 70)]


def make_gif(folder, filename):
    images = sorted(folder.glob("frame_*.jpg"))
    if not images:
        raise RuntimeError(f"Нет кадров для {filename}")
    with Image.open(images[0]) as first:
        first.save(ROOT / filename, format="GIF", save_all=True,
                   append_images=[Image.open(p).convert("RGB") for p in images[1:]],
                   duration=500, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-frame", type=int,
                        help="Проверить только кадр с указанным номером без сборки GIF")
    args = parser.parse_args()
    if not FRAMES:
        raise SystemExit("Нет кадров в frames/. Сначала извлеките их через ffmpeg.")
    frames = FRAMES
    if args.preview_frame is not None:
        if not 1 <= args.preview_frame <= len(FRAMES):
            raise SystemExit(f"Номер кадра должен быть от 1 до {len(FRAMES)}")
        frames = [FRAMES[args.preview_frame - 1]]
    for folder in (DET_DIR, SEG_DIR, DATA_DIR):
        folder.mkdir(exist_ok=True)

    torch.set_num_threads(4)
    det_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    seg_weights = DeepLabV3_ResNet50_Weights.DEFAULT
    print("Загрузка моделей (при первом запуске скачаются веса)...", flush=True)
    detector = fasterrcnn_resnet50_fpn_v2(weights=det_weights,
                                          min_size=640, max_size=960).eval()
    segmenter = deeplabv3_resnet50(weights=seg_weights).eval()
    det_classes = det_weights.meta["categories"]
    seg_classes = seg_weights.meta["categories"]
    seg_transform = seg_weights.transforms()
    log = []

    for index, path in enumerate(frames, 1):
        start = time.perf_counter()
        source = Image.open(path).convert("RGB")
        width, height = source.size
        tensor = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            detections = detector([tensor])[0]
            seg_logits = segmenter(seg_transform(source).unsqueeze(0))["out"]
            seg_logits = F.interpolate(seg_logits, size=(height, width),
                                       mode="bilinear", align_corners=False)
            probabilities, predicted = seg_logits.softmax(1).max(1)
            label_map = predicted[0].to(torch.uint8).numpy()
            certainty = probabilities[0].numpy()
            label_map[certainty < 0.55] = 0
            for class_id in np.unique(label_map):
                if class_id and np.count_nonzero(label_map == class_id) < width * height * 0.002:
                    label_map[label_map == class_id] = 0

        det_image = source.copy()
        draw = ImageDraw.Draw(det_image)
        objects = []
        for box, label, score in zip(detections["boxes"], detections["labels"],
                                     detections["scores"]):
            confidence = float(score)
            if confidence < CONFIDENCE:
                continue
            coords = [round(float(x), 1) for x in box]
            name = det_classes[int(label)]
            objects.append({"box_xyxy": coords, "class": name,
                            "confidence": round(confidence, 4)})
            x1, y1, x2, y2 = coords
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            caption = f"{name} {confidence:.2f}"
            left, top, right, bottom = draw.textbbox((x1, y1), caption)
            text_y = max(0, y1 - (bottom - top) - 6)
            draw.rectangle((x1, text_y, x1 + right - left + 6,
                            text_y + bottom - top + 5), fill="red")
            draw.text((x1 + 3, text_y + 2), caption, fill="white")
        det_image.save(DET_DIR / path.name, quality=88)

        base = np.asarray(source).copy()
        overlay = base.copy()
        present = [int(c) for c in np.unique(label_map) if int(c) != 0]
        for class_id in present:
            color = COLORS[1 + (class_id - 1) % (len(COLORS) - 1)]
            overlay[label_map == class_id] = color
        foreground = label_map != 0
        base[foreground] = (0.55 * base[foreground] +
                            0.45 * overlay[foreground]).astype(np.uint8)
        seg_image = Image.fromarray(base)
        legend = ImageDraw.Draw(seg_image)
        for row, class_id in enumerate(present):
            y = 10 + row * 22
            color = COLORS[1 + (class_id - 1) % (len(COLORS) - 1)]
            legend.rectangle((10, y, 25, y + 15), fill=color)
            legend.rectangle((30, y - 1, 150, y + 17), fill="black")
            legend.text((34, y + 1), seg_classes[class_id], fill="white")
        seg_image.save(SEG_DIR / path.name, quality=88)

        record = {"frame": path.name, "objects": objects,
                  "segmentation_classes": [seg_classes[c] for c in present],
                  "processing_seconds": round(time.perf_counter() - start, 2)}
        (DATA_DIR / f"{path.stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(DATA_DIR / f"{path.stem}_mask.npz", labels=label_map)
        log.append(record)
        print(f"{index}/{len(frames)}: {path.name} — "
              f"{len(objects)} объектов, сегментация: {record['segmentation_classes']}, "
              f"{record['processing_seconds']} с", flush=True)

    if args.preview_frame is not None:
        print("Пробный кадр готов. Проверьте два изображения frame_*.jpg.", flush=True)
        return
    (ROOT / "summary.json").write_text(
        json.dumps({"input": "input.mp4", "frames": len(FRAMES),
                    "fps": 2, "detection_model": "Faster R-CNN ResNet50 FPN V2",
                    "segmentation_model": "DeepLabV3 ResNet50",
                    "confidence_threshold": CONFIDENCE,
                    "total_processing_seconds": round(sum(x["processing_seconds"] for x in log), 2),
                    "results": log}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Сборка GIF...", flush=True)
    make_gif(DET_DIR, "detection.gif")
    make_gif(SEG_DIR, "segmentation.gif")
    print("Готово: detection.gif и segmentation.gif", flush=True)


if __name__ == "__main__":
    main()
