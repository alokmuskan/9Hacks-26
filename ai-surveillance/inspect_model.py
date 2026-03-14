from __future__ import annotations

from insightface.model_zoo import model_zoo


def main() -> None:
    try:
        model = model_zoo.get_model("buffalo_l")
        print(f"Model file: {model.model_file}")
        print(f"Task name: {model.taskname}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
