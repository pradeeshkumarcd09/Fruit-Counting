import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Detect and count apples, carrots, and oranges."
    )
    parser.add_argument("--source", required=True,
                        help="Path to an image or folder of images")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--output", default="predictions",
                        help="Directory for annotated predictions")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "models" / "best.pt"
    model = YOLO(str(model_path))

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=args.output,
        name="produce_detection",
        exist_ok=True,
        verbose=False
    )

    names = model.names
    for result in results:
        counts = {name: 0 for name in names.values()}
        for class_id in result.boxes.cls.int().tolist():
            counts[names[class_id]] += 1

        print(f"\nImage: {Path(result.path).name}")
        for name, count in counts.items():
            print(f"  {name}: {count}")
        print(f"  Total detected: {sum(counts.values())}")

    print(f"\nAnnotated outputs saved under: {args.output}/produce_detection")

if __name__ == "__main__":
    main()
