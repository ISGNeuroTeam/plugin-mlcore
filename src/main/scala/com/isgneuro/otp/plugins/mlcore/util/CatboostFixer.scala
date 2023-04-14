package com.isgneuro.otp.plugins.mlcore.util

import org.apache.spark.ml.{Transformer, linalg}
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}


class CatboostFixer(val uid: String, oldname: String, name: String) extends Transformer with Params
  with HasInputCols with HasOutputCols with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("ColRenamer"), "11", "12")

  def transformSchema(schema: StructType): StructType = schema

  def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    var newDataset = dataset
    if (dataset.columns.contains(oldname)) {
      val vecToArray = udf((xs: linalg.Vector) => xs.toArray)
      newDataset = newDataset
        .withColumn(oldname, vecToArray(newDataset(oldname)))        
        .drop("rawPrediction", "prediction", "features", "label")

    }
    newDataset.toDF()
  }
}

object CatboostFixer extends DefaultParamsReadable[CatboostFixer] {
  override def load(path: String): CatboostFixer = super.load(path)
}


