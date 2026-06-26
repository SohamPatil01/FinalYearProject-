| Model | Weights | Used for | Precision | Recall | mAP@50 | mAP@50-95 | Source |
|---|---|---|---|---|---|---|---|
| Truck detector | truck.pt | Truck restricted-hours | 0.959 | 0.775 | 0.913 | 0.773 | Trained val split |
| Number-plate detector | plate_best.pt | Plate localization (for OCR) | 0.981 | 0.979 | 0.989 | 0.794 | Trained val split |
| Helmet detector | helmet_best.pt | No-helmet rule | 0.813 | 0.777 | 0.841 | 0.364 | Trained val split |
| Triple-seat detector | triple.pt | Triple-riding rule | 0.407 | 0.571 | 0.451 | 0.191 | Trained val split |
| YOLOv10-S | yolov10s.pt | Red-light engine | — | — | — | 0.463 | COCO benchmark (published) |
| YOLOv8-n | yolov8n.pt | No-parking + vehicle/rider scoping | — | — | — | 0.373 | COCO benchmark (published) |
