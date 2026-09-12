from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# SETTINGS
# ============================================================

MASTER_FILE = Path(
    r"D:\OncoLens\processed\master_verified_dataset.csv"
)

OUTPUT_DIR = Path(
    r"D:\OncoLens\processed\stratified_splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(MASTER_FILE)

print("=" * 80)
print("MASTER DATASET")
print("=" * 80)

print("Total images:", len(df))
print("Unique patients:", df["patient_id"].nunique())


# ============================================================
# CREATE PATIENT-LEVEL LABEL SIGNATURE
# ============================================================

def create_signature(labels):
    return " + ".join(sorted(set(labels)))


patient_info = (
    df.groupby("patient_id")
    .agg(
        label_signature=(
            "pathology",
            create_signature
        )
    )
    .reset_index()
)


print("\n" + "=" * 80)
print("PATIENT LABEL SIGNATURE DISTRIBUTION")
print("=" * 80)

print(
    patient_info["label_signature"]
    .value_counts()
)


# ============================================================
# CHECK SIGNATURE COUNTS
# ============================================================

signature_counts = (
    patient_info["label_signature"]
    .value_counts()
)

rare_signatures = signature_counts[
    signature_counts < 3
]

if len(rare_signatures) > 0:

    print("\nWARNING: Rare signatures detected:")
    print(rare_signatures)

    print("\nThese signatures cannot be safely stratified.")

else:

    print("\nAll signatures have enough patients for stratification.")


# ============================================================
# 70% TRAIN / 30% TEMP
# ============================================================

train_patients, temp_patients = train_test_split(
    patient_info,
    test_size=0.30,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=patient_info["label_signature"]
)


# ============================================================
# 15% VALIDATION / 15% TEST
# ============================================================

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=temp_patients["label_signature"]
)


# ============================================================
# GET PATIENT ID SETS
# ============================================================

train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])


# ============================================================
# CREATE IMAGE-LEVEL SPLITS
# ============================================================

train_df = df[
    df["patient_id"].isin(train_ids)
].copy()

val_df = df[
    df["patient_id"].isin(val_ids)
].copy()

test_df = df[
    df["patient_id"].isin(test_ids)
].copy()


# ============================================================
# ADD SPLIT COLUMN
# ============================================================

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


# ============================================================
# VERIFY PATIENT OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("PATIENT OVERLAP CHECK")
print("=" * 80)

print("Train ∩ Val :", len(train_ids & val_ids))
print("Train ∩ Test:", len(train_ids & test_ids))
print("Val ∩ Test  :", len(val_ids & test_ids))


# ============================================================
# SPLIT SIZE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SPLIT SIZES")
print("=" * 80)

summary = pd.DataFrame({
    "split": ["train", "val", "test"],
    "images": [
        len(train_df),
        len(val_df),
        len(test_df)
    ],
    "patients": [
        train_df["patient_id"].nunique(),
        val_df["patient_id"].nunique(),
        test_df["patient_id"].nunique()
    ]
})

print(summary.to_string(index=False))


# ============================================================
# PATHOLOGY DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("PATHOLOGY DISTRIBUTION")
print("=" * 80)

for name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")

    counts = split_df["pathology"].value_counts()
    percentages = (
        split_df["pathology"]
        .value_counts(normalize=True)
        * 100
    )

    for label in sorted(counts.index):

        print(
            f"{label:25s} "
            f"{counts[label]:4d} "
            f"({percentages[label]:5.2f}%)"
        )


# ============================================================
# LESION TYPE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("LESION TYPE DISTRIBUTION")
print("=" * 80)

for name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")

    counts = split_df["lesion_type"].value_counts()
    percentages = (
        split_df["lesion_type"]
        .value_counts(normalize=True)
        * 100
    )

    for lesion in sorted(counts.index):

        print(
            f"{lesion:10s} "
            f"{counts[lesion]:4d} "
            f"({percentages[lesion]:5.2f}%)"
        )


# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "val.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# ============================================================
# SAVE ALL SPLITS
# ============================================================

all_splits = pd.concat([
    train_df,
    val_df,
    test_df
])

all_splits.to_csv(
    OUTPUT_DIR / "all_splits.csv",
    index=False
)


# ============================================================
# SAVE PATIENT SPLIT ASSIGNMENTS
# ============================================================

patient_split = pd.DataFrame({
    "patient_id": (
        list(train_ids)
        + list(val_ids)
        + list(test_ids)
    ),
    "split": (
        ["train"] * len(train_ids)
        + ["val"] * len(val_ids)
        + ["test"] * len(test_ids)
    )
})

patient_split.to_csv(
    OUTPUT_DIR / "patient_split_assignments.csv",
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "val.csv")
print(OUTPUT_DIR / "test.csv")
print(OUTPUT_DIR / "all_splits.csv")
print(OUTPUT_DIR / "patient_split_assignments.csv")

print("\nDONE!")