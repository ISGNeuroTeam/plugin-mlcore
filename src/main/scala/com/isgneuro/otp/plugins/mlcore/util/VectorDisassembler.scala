package com.isgneuro.otp.plugins.mlcore.util

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCols}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Transformer, linalg}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}


class VectorDisassembler(override val uid: String) extends Transformer
  with HasInputCol
  with HasOutputCols
  with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("vecDisassembler"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)


  override def transform(df: Dataset[_]): DataFrame = {
    val vecToArray = udf((xs: linalg.Vector) => xs.toArray)
    // Add an ArrayType column
    val dfArr = df.withColumn("featuresArr", vecToArray(df($(inputCol))))

    // Split array into columns
    val out = $(outputCols).zipWithIndex.foldLeft(dfArr)((acc, c) =>
      acc.withColumn(c._1, col("featuresArr").getItem(c._2))
    )
    out.drop("featuresArr")
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val inputFields = schema.fields
    require($(outputCols).forall { c => !inputFields.exists(_.name == c) },
      s"Output columns ${$(outputCols).mkString(", ")} already exist.")

    val newFields = $(outputCols).foldLeft(schema.fields) { (acc, c) => acc :+ StructField(c, DoubleType) }
    StructType(newFields)
  }
}

object VectorDisassembler extends DefaultParamsReadable[VectorDisassembler] {
  override def load(path: String): VectorDisassembler = super.load(path)
}