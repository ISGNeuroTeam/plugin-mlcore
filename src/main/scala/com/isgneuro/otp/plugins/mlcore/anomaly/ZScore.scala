package com.isgneuro.otp.plugins.mlcore.anomaly

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame}
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils

case class ZScore(fieldsUsed: List[String], properties: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val BY_DEFAULT: List[String] = List[String]()
  val MEDIAN_DEFAULT = "approx_percentile"
  val WINDOW_BEFORE_DEFAULT: Long = -1 * Window.unboundedPreceding
  val WINDOW_AFTER_DEFAULT: Long = Window.unboundedFollowing
  val WITH_MEAN_DEFAULT = false
  val WITH_STD_DEFAULT = false

  /**
   *
   * @return input dataframe with added column for each input feature with computed ZScore
   *
   *         keywords:
   *  - by: Defines the column(s) to group all values by. Default: None.
   *  - window_before: The number of rows before to take into account. Default: unbounded.
   *  - window_after: The number of rows after to take into account. Default: unbounded.
   *  - window: The number of rows before to take into account. With using this, window_after is set to 0. Default: unbounded.
   *  - with_mean: Defines if the output has mean columns. Default: false.
   *  - with_std: Defines if the output has std columns. Default: false.
   */

  def makePrediction(dataFrame: DataFrame): DataFrame = {
    val by = properties.get("by") match {
      case Some(m) => m.split(",").map(_.trim).toList
      case None => BY_DEFAULT
    }
    val withMean = Caster.safeCast[Boolean](
      properties.get("with_mean"),
      WITH_MEAN_DEFAULT,
      sendError(searchId, "The value of parameter 'with_mean' should be of boolean type")
    )
    val withSTD = Caster.safeCast[Boolean](
      properties.get("with_std"),
      WITH_STD_DEFAULT,
      sendError(searchId, "The value of parameter 'with_std' should be of boolean type")
    )
    val window_before = Caster.safeCast[Long](
      properties.get("window_before"),
      Caster.safeCast[Long](
        properties.get("window"),
        WINDOW_BEFORE_DEFAULT,
        sendError(searchId, "The value of parameter 'window' should be of int type")
      ),
      sendError(searchId, "The value of parameter 'window_before' should be of int type")
    )
    val window_after = Caster.safeCast[Long](
      properties.get("window_after"),
      Caster.safeCast[Long](
        properties.get("window"),
        WINDOW_AFTER_DEFAULT,
        sendError(searchId, "The value of parameter 'window' should be of int type")
      ),
      sendError(searchId, "The value of parameter 'window_after' should be of int type")
    )
    val byColumns = Window.partitionBy(by.map(col): _*).rowsBetween(-1 * window_before, window_after)

    def zscore(mean: Column, sd: Column)(x: Column) =
      (x - mean) / sd

    val result = fieldsUsed.foldLeft(dataFrame) {
      case (acc, field) =>
        val withMeanStdev = acc.withColumn("mean_" + field, avg(field).over(byColumns))
          .withColumn("stdev_" + field, stddev(field).over(byColumns))
        val withZScore = withMeanStdev.withColumn("zscore_" + field,
          zscore(withMeanStdev("mean_" + field), withMeanStdev("stdev_" + field))(acc(field)))
        val meanCols = withZScore.columns.filter(_.matches("mean_.*"))
        val stdCols = withZScore.columns.filter(_.matches("stdev_.*"))
        val meanCheck = if (!withMean) withZScore.drop(meanCols: _*) else withZScore
        if (!withSTD) meanCheck.drop(stdCols: _*) else meanCheck
    }
    result
  }
}

object ZScore extends ApplyModel {
  override def apply(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String], targetName: Option[String], keywords: Map[String, String], utils: PluginUtils): DataFrame => DataFrame =
    ZScore(featureCols, keywords, searchId, utils).makePrediction
}
