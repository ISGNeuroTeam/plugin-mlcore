package com.isgneuro.otp.plugins.mlcore.stat

import com.typesafe.config.Config
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.functions.{col, desc, explode, expr}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils


case class Correlation(featureCols: List[String], keywords: Map[String, String], id: Int, utils: PluginUtils) {

  import utils._

  val METHOD_DEFAULT = "pearson"
  val SUPPORTED_METHODS = List("pearson", "spearman")

  /**
   *
   * @param df is input DataFrame
   * @return correlation matrix
   *
   *         keywords:
   *  - method: The correlation method, pearson or spearman. Default: pearson
   */

  def makePrediction(df: DataFrame): DataFrame = {
    val method = keywords.get("method") match {
      case Some(x) => if (SUPPORTED_METHODS.contains(x)) x else sendError(id, "The value of parameter 'method' should be 'pearson' or 'spearman'")
      case None => METHOD_DEFAULT
    }

    val output = keywords.getOrElse("output", "wide")

    val featuresName = s"__features__"
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol(featuresName)
    val transformedDf = assembler.transform(df)

    val Row(coeff1: Matrix) = org.apache.spark.ml.stat.Correlation.corr(transformedDf, "__features__", method).head
    val matrixRows = coeff1.rowIter.toSeq.map(_.toArray)
    val matrixRowsWithNames = featureCols.zip(matrixRows).map(item => item._1 +: item._2)

    val schema = StructField("xCol", StringType, nullable=false) :: featureCols.map(x => StructField(x, DoubleType, nullable = true))
    val result = spark.createDataFrame(
      spark.sparkContext.parallelize(matrixRowsWithNames.map(Row.fromSeq(_))),
      StructType(schema)
    ).select("xCol", featureCols: _*)

    output match {
      case "long" => makeLongTable(result, "xCol", featureCols)
      case "wide" => result
      case _ => sendError(id, "Incorrect output type")
    }
  }

  def makeLongTable(df: DataFrame, fixedCol: String = "xCol", otherCols: List[String]): DataFrame = {
    otherCols.foldLeft(df) {
      (accum, colname) =>
        accum.withColumn(colname, expr(s"""array("$colname", ${colname})"""))
    }.withColumn("arr", expr(s"""array(${otherCols.mkString(", ")})"""))
      .select(fixedCol, "arr")
      .withColumn("arr", explode(col("arr")))
      .withColumn("yCol", col("arr").getItem(0))
      .withColumn("corr", col("arr").getItem(1).cast("double"))
      .select(fixedCol, "yCol", "corr")
      .orderBy(desc("xCol"), desc("yCol"))
  }
}

object Correlation extends ApplyModel {
  override def apply(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String], targetName: Option[String], keywords: Map[String, String], utils: PluginUtils): DataFrame => DataFrame =
    Correlation(featureCols, keywords, searchId, utils).makePrediction
}

