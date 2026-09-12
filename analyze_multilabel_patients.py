from pathlib import Path
import pandas as pd

# ============================================================
# PATH
# ============================================================

MASTER_FILE = Path(
    r"D:\OncoLens\processed\master_verified_dataset.csv"
)

OUTPUT_FILE = Path(
    r"D:\OncoLens\processed\multilabel_patients_analysis.csv"
)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(MASTER_FILE)


# ============================================================
# FIND PATIENTS WITH MULTIPLE LABELS
# ============================================================

label_counts = (
    df.groupby("patient_id")["pathology"]
    .nunique()
)

multilabel_patients = label_counts[
    label_counts > 1
].index

multilabel_df = df[
    df["patient_id"].isin(multilabel_patients)
].copy()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 80)
print("MULTI-LABEL PATIENT ANALYSIS")
print("=" * 80)

print(f"\nTotal multi-label patients: {len(multilabel_patients)}")

print(f"Total records belonging to these patients: {len(multilabel_df)}")


# ============================================================
# LABEL COMBINATIONS
# ============================================================

patient_combinations = (
    multilabel_df
    .groupby("patient_id")["pathology"]
    .apply(lambda x: " + ".join(sorted(set(x))))
)

combination_counts = patient_combinations.value_counts()

print("\n" + "=" * 80)
print("PATHOLOGY LABEL COMBINATIONS")
print("=" * 80)

print(combination_counts)


# ============================================================
# LESION TYPE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("MULTI-LABEL RECORDS BY LESION TYPE")
print("=" * 80)

print(
    multilabel_df["lesion_type"]
    .value_counts()
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("EXAMPLE MULTI-LABEL PATIENTS")
print("=" * 80)

for patient_id in list(multilabel_patients)[:15]:

    patient_data = df[
        df["patient_id"] == patient_id
    ]

    print("\n" + "-" * 80)
    print(f"PATIENT: {patient_id}")

    print(
        patient_data[
            [
                "image_id",
                "pathology",
                "lesion_type",
                "original_source"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

multilabel_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 80)
print("ANALYSIS SAVED")
print("=" * 80)

print(OUTPUT_FILE)