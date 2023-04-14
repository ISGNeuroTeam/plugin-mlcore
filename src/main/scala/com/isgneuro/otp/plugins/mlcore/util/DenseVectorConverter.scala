package com.isgneuro.otp.plugins.mlcore.util

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._ 


class DenseVectorConverter(val uid: String) extends Transformer with Params
  with HasInputCols with HasOutputCols with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("denseVectorConverter"))

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  def validateAndTransformSchema(schema: StructType): StructType = {
    require($(inputCols).length == $(inputCols).distinct.length, s"inputCols contains" +
      s" duplicates: (${$(inputCols).mkString(", ")})")
    require($(outputCols).length == $(outputCols).distinct.length, s"outputCols contains" +
      s" duplicates: (${$(outputCols).mkString(", ")})")
    require($(inputCols).length == $(outputCols).length, s"inputCols(${$(inputCols).length})" +
      s" and outputCols(${$(outputCols).length}) should have the same length")

    $(inputCols).zip($(outputCols)).foldLeft(schema) { (schema, inOutCol) =>
      val inputField = schema(inOutCol._1)
      require(inputField.dataType == SQLDataTypes.VectorType, s"Expected dtatype of input col: ${inputField.name} as " +
        s"vector but found ${inputField.dataType}")
      schema.add(inOutCol._2, inputField.dataType, inputField.nullable, inputField.metadata)
    }
  }

  def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val sparseToDense =
      udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)
    $(inputCols).zip($(outputCols)).foldLeft(dataset.toDF()) { (df, inputColOutputCol) =>

      df.withColumn(inputColOutputCol._2,
        sparseToDense(col(inputColOutputCol._1)));
    }
  }
}

object DenseVectorConverter extends DefaultParamsReadable[DenseVectorConverter] {
  override def load(path: String): DenseVectorConverter = super.load(path)
}
