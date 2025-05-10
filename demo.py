# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)
<<<<<<< HEAD
# COMMAND ----------
=======
# COMMAND ----------
>>>>>>> 08790bf (bugfix)
