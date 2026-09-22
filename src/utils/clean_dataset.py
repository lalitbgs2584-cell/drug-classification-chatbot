from pyspark.sql import functions as F

def clean_dataset(df, include_rag_doc=True):    
    # Find columns based on their prefixes
    side_effect_cols = [c for c in df.columns if c.startswith("sideEffect")]
    use_cols = [c for c in df.columns if c.startswith("use")]
    sub_cols = [c for c in df.columns if c.startswith("substitute")]

    # Clean nulls, whitespace-only, "NA", "NaN", "null", and empty strings
    clean_col = lambda c: F.when(
        F.col(c).isNull() |
        F.lower(F.trim(F.col(c))).isin("", "na", "n/a", "nan", "null", "none"),
        None
    ).otherwise(F.trim(F.col(c)))

    side_effects = F.concat_ws(", ", *[clean_col(c) for c in side_effect_cols])
    uses = F.concat_ws(", ", *[clean_col(c) for c in use_cols])
    subs = F.concat_ws(", ", *[clean_col(c) for c in sub_cols])

    # Rename original columns to lowercase with underscores (snake_case)
    column_renames = {
        "name": "drug_name",
        "Chemical Class": "chemical_class",
        "Habit Forming": "habit_forming",
        "Therapeutic Class": "therapeutic_class",
        "Action Class": "action_class",
    }
    for old_col, new_col in column_renames.items():
        if old_col in df.columns:
            df = df.withColumnRenamed(old_col, new_col)

    # Clean empty / NA values in the remaining metadata columns as well
    for col_name in ["chemical_class", "habit_forming", "therapeutic_class", "action_class"]:
        if col_name in df.columns:
            df = df.withColumn(col_name, clean_col(col_name))

    # Add consolidated single columns (set to None if empty after join)
    df = df.withColumn("uses", F.when(F.trim(uses) == "", None).otherwise(uses)) \
           .withColumn("side_effects", F.when(F.trim(side_effects) == "", None).otherwise(side_effects)) \
           .withColumn("substitutes", F.when(F.trim(subs) == "", None).otherwise(subs))

    # Build rag_doc
    if include_rag_doc:
        df = df.withColumn(
            "rag_doc",
            F.concat(
                F.lit("Medicine: "),
                F.coalesce(F.col("drug_name"), F.lit("NA")),
                F.lit("\nUses: "),
                F.coalesce(F.col("uses"), F.lit("Not specified")),
                F.lit("\nSide Effects: "),
                F.coalesce(F.col("side_effects"), F.lit("Not specified")),
                F.lit("\nSubstitutes: "),
                F.coalesce(F.col("substitutes"), F.lit("None listed")),
                F.lit("\nChemical Class: "),
                F.coalesce(F.col("chemical_class"), F.lit("NA")),
                F.lit("\nTherapeutic Class: "),
                F.coalesce(F.col("therapeutic_class"), F.lit("NA")),
                F.lit("\nHabit Forming: "),
                F.coalesce(F.col("habit_forming"), F.lit("NA"))
            )
        )

    # Reorder to the exact requested columns in lowercase and _ separated
    target_columns = [
        "id",
        "drug_name",
        "uses",
        "side_effects",
        "substitutes",
        "chemical_class",
        "habit_forming",
        "therapeutic_class",
        "action_class"
    ]
    if include_rag_doc:
        target_columns.append("rag_doc")

    final_cols = [c for c in target_columns if c in df.columns]
    df = df.select(*final_cols)

    # Deduplicate by drug_name, uses, and side_effects
    initial_count = df.count()
    df = df.dropDuplicates(subset=["drug_name", "uses", "side_effects"])
    dedup_count = df.count()
    print(f"Deduplication: {initial_count} -> {dedup_count}")

    return df