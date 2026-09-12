from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\OncoLens")

DATASET_DIR = (
    PROJECT_DIR
    / "Data"
    / "CBIS-DDSM Breast Cancer Image Dataset"
)

CSV_DIR = DATASET_DIR / "csv"
JPEG_DIR = DATASET_DIR / "jpeg"


# ============================================================
# 1. VERIFY DIRECTORIES
# ============================================================

print("=" * 60)
print("DIRECTORY CHECK")
print("=" * 60)

directories = {
    "Project": PROJECT_DIR,
    "Dataset": DATASET_DIR,
    "CSV": CSV_DIR,
    "JPEG": JPEG_DIR
}

for name, path in directories.items():
    print(f"{name}: {path}")
    print(f"Exists: {path.exists()}\n")


# ============================================================
# 2. LIST CSV FILES
# ============================================================

print("=" * 60)
print("CSV FILES")
print("=" * 60)

csv_files = sorted(CSV_DIR.glob("*.csv"))

for file in csv_files:
    print(file.name)

print(f"\nTotal CSV files: {len(csv_files)}")


# ============================================================
# 3. INSPECT JPEG STRUCTURE
# ============================================================

print("\n" + "=" * 60)
print("JPEG DIRECTORY")
print("=" * 60)

jpeg_items = list(JPEG_DIR.iterdir())

print(f"Total top-level items: {len(jpeg_items)}")

print("\nFirst 10 items:")

for item in jpeg_items[:10]:
    item_type = "Directory" if item.is_dir() else "File"
    print(f"{item.name} | {item_type}")


# ============================================================
# 4. COUNT ACTUAL JPEG FILES
# ============================================================

print("\n" + "=" * 60)
print("COUNTING JPEG FILES")
print("=" * 60)

jpeg_count = 0
first_files = []

for file in JPEG_DIR.rglob("*.jpg"):
    jpeg_count += 1

    if len(first_files) < 5:
        first_files.append(file)

print(f"Total .jpg files found: {jpeg_count}")

print("\nFirst 5 JPEG files:")
for file in first_files:
    print(file)
