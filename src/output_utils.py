from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "csv"


def csv_output_path(path):
    path = Path(path)

    if not path.is_absolute() and path.parent == Path("."):
        path = CSV_OUTPUT_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def csv_input_path(path):
    path = Path(path)

    if path.exists() or path.is_absolute() or path.parent != Path('.'):
        return path

    output_path = CSV_OUTPUT_DIR / path
    return output_path if output_path.exists() else path



