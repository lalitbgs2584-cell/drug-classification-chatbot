import pandas as pd
from pyspark.sql import SparkSession
from src.utils import load_dataset
from src.utils.clean_dataset import clean_dataset


spark = SparkSession.builder \
    .appName("DrugClassificationApp") \
    .getOrCreate()

load_dataset.download_file()
df = spark.read.option("header", "true").option("escape", "\"").csv("src/drugs.csv")
df = clean_dataset(df)

print("\n--- Consolidated DataFrame Schema ---")
df.printSchema()

df.select("id", "drug_name", "uses", "side_effects", "substitutes").show(1, truncate=False)

output_csv_path = "src/drugs_cleaned.csv"
print(f"Generating CSV file at '{output_csv_path}'...")
df.toPandas().to_csv(output_csv_path, index=False)
print(f"CSV file generated successfully: '{output_csv_path}'")

spark.stop()