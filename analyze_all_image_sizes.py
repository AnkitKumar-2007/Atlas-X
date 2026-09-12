import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter

# ============================================================
# PATHS
# ============================================================

DATASET_CSV = Path(
    r"D:\OncoLens\processed\stratified_splits\all_splits.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATASET_CSV)

print(f"Total records: {len(df)}")

# ============================================================
# ANALYZE ALL IMAGE DIMENSIONS
# ============================================================

widths = []
heights = []
sizes = []
aspect_ratios = []
errors = []

print("\n" + "=" * 70)
print("ANALYZING ALL IMAGE DIMENSIONS")
print("=" * 70)

for i, row in df.iterrows():

    image_path = row["image_path_local"]

    try:
        with Image.open(image_path) as img:
            width, height = img.size

        widths.append(width)
        heights.append(height)
        sizes.append((width, height))
        aspect_ratios.append(width / height)

    except Exception as e:
        errors.append({
            "index": i,
            "image_path": image_path,
            "error": str(e)
        })

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(df)} images...")

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("IMAGE SIZE ANALYSIS RESULTS")
print("=" * 70)

print(f"\nSuccessfully analyzed: {len(widths)}")
print(f"Errors: {len(errors)}")

if widths:

    width_series = pd.Series(widths)
    height_series = pd.Series(heights)
    ratio_series = pd.Series(aspect_ratios)

    print("\n" + "-" * 70)
    print("WIDTH STATISTICS")
    print("-" * 70)

    print(f"Minimum: {width_series.min()}")
    print(f"Maximum: {width_series.max()}")
    print(f"Mean:    {width_series.mean():.2f}")
    print(f"Median:  {width_series.median():.2f}")

    print("\n" + "-" * 70)
    print("HEIGHT STATISTICS")
    print("-" * 70)

    print(f"Minimum: {height_series.min()}")
    print(f"Maximum: {height_series.max()}")
    print(f"Mean:    {height_series.mean():.2f}")
    print(f"Median:  {height_series.median():.2f}")

    print("\n" + "-" * 70)
    print("ASPECT RATIO (WIDTH / HEIGHT)")
    print("-" * 70)

    print(f"Minimum: {ratio_series.min():.3f}")
    print(f"Maximum: {ratio_series.max():.3f}")
    print(f"Mean:    {ratio_series.mean():.3f}")
    print(f"Median:  {ratio_series.median():.3f}")

    print("\nPercentiles:")

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(
            f"{p:>2}th percentile → "
            f"Width: {width_series.quantile(p/100):.0f}, "
            f"Height: {height_series.quantile(p/100):.0f}"
        )

    # ========================================================
    # MOST COMMON IMAGE SIZES
    # ========================================================

    print("\n" + "-" * 70)
    print("TOP 20 MOST COMMON IMAGE DIMENSIONS")
    print("-" * 70)

    size_counts = Counter(sizes)

    for size, count in size_counts.most_common(20):
        print(f"{size[0]} × {size[1]} : {count} images")

# ============================================================
# ERRORS
# ============================================================

if errors:

    print("\n" + "=" * 70)
    print("ERRORS")
    print("=" * 70)

    error_df = pd.DataFrame(errors)

    ERROR_OUTPUT = Path(
        r"D:\OncoLens\processed\image_dimension_errors.csv"
    )

    error_df.to_csv(ERROR_OUTPUT, index=False)

    print(f"\nSaved errors to:")
    print(ERROR_OUTPUT)

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "total_images": len(df),
    "successfully_analyzed": len(widths),
    "errors": len(errors),
}

if widths:
    summary.update({
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": sum(widths) / len(widths),
        "median_width": pd.Series(widths).median(),

        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": sum(heights) / len(heights),
        "median_height": pd.Series(heights).median(),
    })

SUMMARY_OUTPUT = Path(
    r"D:\OncoLens\processed\image_size_analysis.csv"
)

pd.DataFrame([summary]).to_csv(SUMMARY_OUTPUT, index=False)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(SUMMARY_OUTPUT)

print("\nDONE!")