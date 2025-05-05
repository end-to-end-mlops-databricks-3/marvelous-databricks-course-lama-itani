# Databricks notebook source
# MAGIC %pip install /Volumes/dev/acikgozm_c3/packages/house_price-latest-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------
from loguru import logger
import yaml
import sys
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.data_processor import DataProcessor


logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO")  # Add handler to stdout

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

print("config", config)
# COMMAND ----------

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv", header=True, inferSchema=True
).toPandas()


# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
# COMMAND ----------
