import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.config_manager import Config, DataConfig

def clean_data(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values and standardizing column names.

    Parameters:
    config (Config): The configuration object containing cleaning settings.
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    # Standardize column names
    df.columns = [col.strip().lower().replace(" ", "_").replace("temparature", "temperature") for col in df.columns]

    # Keep only: id, features, target (NOT metadata like _source, _ingestion_timestamp)
    keep_cols = ['id'] + config.data.features.numerical + config.data.features.categorical + [config.target_column]
    for col in df.columns:
        if col not in keep_cols:
            df = df.drop(columns=[col])
    
    df = df.dropna()  # Drop rows with any missing values

    return df

def feature_engineer_data(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the input DataFrame.

    Parameters:
    config (Config): The configuration object containing feature settings.
    df (pd.DataFrame): The input DataFrame to be feature engineered.

    Returns:
    pd.DataFrame: The DataFrame with engineered features.
    """
    # Placeholder for feature engineering logic
    return df

def ingest_data(config: Config, file_path: str, source: str = "local") -> None:
    """
    Ingest data from a specified file path into a Spark DataFrame.
    """
    data_config = config.data
    spark = SparkSession.builder.appName("DataIngestion").getOrCreate()
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower().replace(" ", "_").replace("temparature", "temperature") for col in df.columns]
    
    # === BRONZE: Raw table with metadata ===
    ingested_raw_df = spark.createDataFrame(df)
    ingested_raw_df = (
        ingested_raw_df
        .withColumn("_source", F.lit(source))
        .withColumn("_ingestion_timestamp", F.current_timestamp())
    )
    
    try:
        raw_df = spark.read.table(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.raw_table_name}')
        ingested_raw_df.write.mode('append').option("mergeSchema", "true").saveAsTable(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.raw_table_name}')
    except Exception:
        ingested_raw_df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.raw_table_name}')

    # === SILVER: Clean table with metadata ===
    ingested_cleaned_df = spark.createDataFrame(clean_data(config, df))
    ingested_cleaned_df = (
        ingested_cleaned_df
        .withColumn("_source", F.lit(source))
        .withColumn("_ingestion_timestamp", F.current_timestamp())
    )
    
    # Silver keeps: meta_features (id, _source, _ingestion_timestamp) + features + target
    silver_keep_cols = (
        data_config.meta_features + 
        data_config.features.numerical + 
        data_config.features.categorical + 
        [config.target_column]
    )
    
    try:
        cleaned_df = spark.read.table(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.intermediate_clean_table}')
        
        # Drop columns NOT in keep list
        for col in cleaned_df.columns:
            if col not in silver_keep_cols:
                cleaned_df = cleaned_df.drop(col)
        
        cleaned_df = cleaned_df.union(ingested_cleaned_df)
        cleaned_df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.intermediate_clean_table}')
    except Exception:
        ingested_cleaned_df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.intermediate_clean_table}')

    # === GOLD: Feature table (only id + features + target, no other metadata) ===
    feature_df = feature_engineer_data(config, df)
    ingested_featured_df = spark.createDataFrame(feature_df)
    
    # Gold keeps: id + features + target (no _source, _ingestion_timestamp)
    gold_keep_cols = (
        ['id'] + 
        data_config.features.numerical + 
        data_config.features.categorical + 
        [config.target_column]
    )
    
    try:
        featured_df = spark.read.table(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.features_table_name}')
        
        for col in featured_df.columns:
            if col not in gold_keep_cols:
                featured_df = featured_df.drop(col)
        
        featured_df = featured_df.union(ingested_featured_df)
        featured_df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.features_table_name}')
    except Exception:
        ingested_featured_df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(f'{data_config.catalog_name}.{data_config.schema_name}.{data_config.features_table_name}')