package com.isgneuro.otp.plugins.mlcore.util

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{ DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}


class ColRenamer(val uid: String, oldname: String, name: String) extends Transformer with Params
  with HasInputCols with HasOutputCols with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("ColRenamer"), "11", "12")

  def transformSchema(schema: StructType): StructType = schema

  def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    var newDataset = dataset
    newDataset.toDF().show()
    if (dataset.columns.contains(oldname)) {
      newDataset = dataset.withColumnRenamed(oldname, name)
    }
    newDataset.toDF()
  }
}
