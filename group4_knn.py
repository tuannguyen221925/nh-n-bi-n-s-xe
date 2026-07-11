import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


CHAR_SIZE = (20, 30)  # width, height
DEFAULT_CHAR_DIR = Path("data/char_images")
DEFAULT_MODEL_DIR = Path("model_data")
LABELS_CSV = "labels.csv"
CLASSIFICATIONS_TXT = "classifications.txt"
FLATTENED_IMAGES_TXT = "flattened_images.txt"
KNN_MODEL = "knn_model.yml"
EVALUATION_REPORT = "evaluation_report.txt"
CONFUSION_MATRIX = "confusion_matrix.csv"


def iter_char_images(char_dir):
    return sorted(
        p for p in Path(char_dir).glob("*/*.png")
        if p.is_file() and p.name.startswith("char_")
    )


def preprocess_char(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    img = cv2.resize(img, CHAR_SIZE)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    _, binary = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # KNN chi can vector 600 pixel: 20 x 30.
    return binary.reshape(1, -1).astype(np.float32)


def init_labels(char_dir, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    labels_path = model_dir / LABELS_CSV

    existing = {}
    if labels_path.exists():
        with labels_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["image_path"]] = row.get("label", "")

    rows = []
    for image_path in iter_char_images(char_dir):
        key = image_path.as_posix()
        rows.append({"image_path": key, "label": existing.get(key, "")})

    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Da tao/cap nhat {labels_path} voi {len(rows)} anh ky tu.")
    print("Hay dien cot label bang ky tu that: 0-9, A-Z.")


def load_labeled_samples(model_dir):
    import numpy as np

    labels_path = Path(model_dir) / LABELS_CSV
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Chua co {labels_path}. Chay: python group4_knn.py init-labels"
        )

    samples = []
    labels = []
    paths = []

    with labels_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "").strip().upper()
            image_path = Path(row["image_path"])

            if not label:
                continue

            if len(label) != 1:
                raise ValueError(f"Nhan phai la 1 ky tu tai {image_path}: {label}")

            samples.append(preprocess_char(image_path))
            labels.append(ord(label))
            paths.append(image_path)

    if not samples:
        raise ValueError("Chua co anh nao duoc gan nhan trong labels.csv.")

    flattened = np.vstack(samples).astype(np.float32)
    classifications = np.array(labels, dtype=np.float32).reshape(-1, 1)
    return flattened, classifications, paths


def stratified_split(y, test_size=0.25, seed=42):
    import numpy as np

    rng = np.random.default_rng(seed)
    groups = defaultdict(list)

    for idx, label in enumerate(y.ravel().astype(int)):
        groups[label].append(idx)

    train_idx = []
    test_idx = []

    for _, indices in groups.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            train_idx.extend(indices.tolist())
            continue

        n_test = max(1, int(round(len(indices) * test_size)))
        n_test = min(n_test, len(indices) - 1)
        test_idx.extend(indices[:n_test].tolist())
        train_idx.extend(indices[n_test:].tolist())

    return np.array(train_idx), np.array(test_idx)


def train_knn(flattened, classifications, k):
    import cv2

    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(k)
    knn.train(flattened, cv2.ml.ROW_SAMPLE, classifications)
    return knn


def evaluate(knn, x_test, y_test, k):
    import numpy as np

    if len(x_test) == 0:
        return None, []

    _, results, _, _ = knn.findNearest(x_test, k=k)
    y_true = y_test.ravel().astype(int)
    y_pred = results.ravel().astype(int)
    accuracy = float(np.mean(y_true == y_pred))
    rows = [(chr(t), chr(p)) for t, p in zip(y_true, y_pred)]
    return accuracy, rows


def write_evaluation(model_dir, accuracy, confusion_rows, label_counts, k):
    model_dir = Path(model_dir)
    report_path = model_dir / EVALUATION_REPORT
    matrix_path = model_dir / CONFUSION_MATRIX

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Bao cao danh gia KNN - Nhom 4\n")
        f.write(f"K = {k}\n")
        f.write(f"So mau da gan nhan = {sum(label_counts.values())}\n")
        f.write("Phan bo nhan:\n")
        for label, count in sorted(label_counts.items()):
            f.write(f"- {label}: {count}\n")

        if accuracy is None:
            f.write("Chua du mau test de tinh accuracy.\n")
        else:
            f.write(f"Accuracy = {accuracy:.4f}\n")
            errors = [(t, p) for t, p in confusion_rows if t != p]
            f.write(f"So mau sai = {len(errors)}\n")
            if errors:
                f.write("Cac loi nhan dien:\n")
                for true_label, pred_label in errors:
                    f.write(f"- {true_label} -> {pred_label}\n")

    labels = sorted({x for row in confusion_rows for x in row})
    counts = Counter(confusion_rows)

    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for true_label in labels:
            writer.writerow(
                [true_label, *[counts[(true_label, pred_label)] for pred_label in labels]]
            )

    print(f"Da ghi {report_path}")
    print(f"Da ghi {matrix_path}")


def train_command(args):
    import numpy as np

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    flattened, classifications, _ = load_labeled_samples(model_dir)
    labels_as_chars = [chr(x) for x in classifications.ravel().astype(int)]
    label_counts = Counter(labels_as_chars)

    if len(label_counts) < 2:
        raise ValueError("Can it nhat 2 lop ky tu khac nhau de danh gia KNN.")

    np.savetxt(model_dir / FLATTENED_IMAGES_TXT, flattened)
    np.savetxt(model_dir / CLASSIFICATIONS_TXT, classifications)

    train_idx, test_idx = stratified_split(
        classifications,
        test_size=args.test_size,
        seed=args.seed,
    )

    knn = train_knn(flattened[train_idx], classifications[train_idx], args.k)
    accuracy, confusion_rows = evaluate(
        knn,
        flattened[test_idx],
        classifications[test_idx],
        args.k,
    )

    full_knn = train_knn(flattened, classifications, args.k)
    full_knn.save(str(model_dir / KNN_MODEL))
    write_evaluation(model_dir, accuracy, confusion_rows, label_counts, args.k)

    print(f"Da luu {model_dir / CLASSIFICATIONS_TXT}")
    print(f"Da luu {model_dir / FLATTENED_IMAGES_TXT}")
    print(f"Da luu {model_dir / KNN_MODEL}")
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f}")


def predict_folder(args):
    import cv2

    model_path = Path(args.model_dir) / KNN_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"Chua co model: {model_path}")

    knn = cv2.ml.KNearest_load(str(model_path))
    chars = []

    for image_path in sorted(Path(args.folder).glob("char_*.png")):
        sample = preprocess_char(image_path)
        _, result, _, _ = knn.findNearest(sample, k=args.k)
        chars.append(chr(int(result[0][0])))

    print("".join(chars))


def main():
    parser = argparse.ArgumentParser(
        description="Nhom 4: gan nhan, train KNN va danh gia nhan dien ky tu bien so."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init-labels")
    p_init.add_argument("--char-dir", default=DEFAULT_CHAR_DIR)
    p_init.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p_init.set_defaults(func=lambda args: init_labels(args.char_dir, args.model_dir))

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p_train.add_argument("--k", type=int, default=3)
    p_train.add_argument("--test-size", type=float, default=0.25)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=train_command)

    p_predict = subparsers.add_parser("predict-folder")
    p_predict.add_argument("folder")
    p_predict.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p_predict.add_argument("--k", type=int, default=3)
    p_predict.set_defaults(func=predict_folder)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        raise SystemExit(f"Loi: {exc}") from exc


if __name__ == "__main__":
    main()
