from pathlib import Path
import shutil

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "src/models/YOLO/yolo11n.pt"
CALIBRATION_YAML = PROJECT_DIR / "src/datasets/calibration/calibration.yaml"
OUTPUT_PATH = PROJECT_DIR / "src/models/YOLO/yolo11n_int8.engine"


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} 파일을 찾을 수 없습니다: {path}")


def main() -> None:
    require_file(MODEL_PATH, "YOLO 모델")
    require_file(CALIBRATION_YAML, "Calibration YAML")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"기존 INT8 Engine을 덮어쓰지 않습니다: {OUTPUT_PATH}\n"
            "기존 파일을 보관하거나 삭제한 뒤 다시 실행하세요."
        )

    print(f"Model:       {MODEL_PATH}")
    print(f"Calibration: {CALIBRATION_YAML}")
    print("TensorRT INT8 Engine 생성을 시작합니다.")

    model = YOLO(str(MODEL_PATH))
    engine_path = Path(
        model.export(
            format="engine",
            imgsz=640,
            int8=True,
            data=str(CALIBRATION_YAML),
            batch=1,
            dynamic=False,
            device=0,
            nms=False,
        )
    ).resolve()

    require_file(engine_path, "생성된 TensorRT Engine")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if engine_path != OUTPUT_PATH:
        shutil.move(str(engine_path), str(OUTPUT_PATH))

    print(f"INT8 Engine 생성 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
