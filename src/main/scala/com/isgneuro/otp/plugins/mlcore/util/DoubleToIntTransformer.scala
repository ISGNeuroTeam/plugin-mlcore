package com.isgneuro.otp.plugins.mlcore.util

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.param.{Param, ParamMap}  
import org.apache.spark.sql.types._         


class DoubleToIntTransformer(override val uid: String)
    extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("TypeCasterTransformer"))
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def getOutputCol: String = getOrDefault(outputCol)

  val inputCol = new Param[String](this, "inputCol", "input column")
  val outputCol = new Param[String](this, "outputCol", "output column")

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outCol = extractParamMap.getOrElse(outputCol, "output")
    val inCol = extractParamMap.getOrElse(inputCol, "input")

    dataset.withColumn(outCol, col(inCol).cast(IntegerType))
  }

  override def copy(extra: ParamMap): DoubleToIntTransformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = schema
}
