import os
import pandas as pd


def sample_by_category(
    csv_path: str = "src/drugs_cleaned.csv",
    category_col: str = "therapeutic_class",
    max_per_category: int = 400,
    random_state: int = 42,
    output_path: str = "src/drugs_sampled.csv"
) -> pd.DataFrame:
    """
    Performs stratified sampling across category_col, taking up to max_per_category rows per group.
    
    Args:
        csv_path: Path to the cleaned CSV file.
        category_col: Categorical column to group and sample by (e.g. 'therapeutic_class', 'action_class').
        max_per_category: Maximum records to sample per category group.
        random_state: Random seed for reproducibility.
        output_path: Destination path for the sampled CSV.
    """
    if not os.path.exists(csv_path):
        # Fallback to drugs_with_rag.csv or drugs.csv if cleaned isn't generated yet
        if os.path.exists("src/drugs_with_rag.csv"):
            csv_path = "src/drugs_with_rag.csv"
        elif os.path.exists("src/drugs.csv"):
            csv_path = "src/drugs.csv"
        else:
            raise FileNotFoundError(f"Could not find dataset at '{csv_path}'.")

    print(f"Loading '{csv_path}' for stratified sampling...")
    df = pd.read_csv(csv_path, low_memory=False)

    if category_col not in df.columns:
        raise ValueError(f"Column '{category_col}' not found in dataframe. Available: {list(df.columns)}")

    # Group by category and sample up to max_per_category per group
    sampled_df = df.groupby(category_col, group_keys=False, dropna=False).apply(
        lambda x: x.sample(min(len(x), max_per_category), random_state=random_state)
    ).reset_index(drop=True)

    category_count = df[category_col].nunique(dropna=False)
    print(f"Sampled {len(sampled_df):,} rows from {len(df):,} total records across {category_count} '{category_col}' groups.")

    if output_path:
        sampled_df.to_csv(output_path, index=False)
        print(f"Saved sampled dataset to '{output_path}'")

    return sampled_df


if __name__ == "__main__":
    sample_by_category()
