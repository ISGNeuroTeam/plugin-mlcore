package com.isgneuro.otp.plugins.mlcore.anomaly

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils

case class MAD(fieldsUsed: List[String], properties: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  private val BY_DEFAULT: Seq[String] = List[String]()
  private val MEDIAN_DEFAULT = "approx_percentile"
  private val WINDOW_BEFORE_DEFAULT = Window.unboundedPreceding
  private val WINDOW_AFTER_DEFAULT = Window.unboundedFollowing

  /**
   *
   * @return input dataframe with added column for each input feature with computed Median Absolute Deviation
   * @see https://en.wikipedia.org/wiki/Median_absolute_deviation
   *
   *      keywords:
   *  - by: Defines the column(s) to group all values by. Default: None.
   *  - window_before: The number of rows before to take into account. Default: unbounded. Out of use now.
   *  - window_after: The number of rows after to take into account. Default: unbounded. Out of use now.
   *  - window: The number of rows before to take into account. With using this, window_after is set to 0. Default: unbounded. Out of use now.
   *  - median: Defines if the median should be computed exactly or approximately. Default: approx.
   */

  def makePrediction(dataFrame: DataFrame): DataFrame = {

    val by = properties.get("by") match {
      case Some(m) => m.split(",").map(_.trim).toList
      case None => BY_DEFAULT
    }
    val median = properties.get("median") match {
      case Some(m) => if (m == "approx") "approx_percentile" else if (m == "exact") "percentile"
      else sendError(searchId, "The value of parameter 'median' should be 'exact' or 'approx'")
      case None => MEDIAN_DEFAULT
    }
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

    // make partition by columns "by" in given window
    // window, window_before, window_after is out of use by now, but MAD in window should be implemented
    val byColumns = Window.partitionBy(by.map(col): _*).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    val result = fieldsUsed.foldLeft(dataFrame) {
      case (acc, field) =>
        val withMedian = acc.withColumn(field + "_median", expr(s"$median($field, array(0.5), 100)").over(byColumns)(0))
        val withDiff = withMedian.withColumn(field + "_diff", abs(withMedian(field) - withMedian(field + "_median")))
        withDiff.withColumn(field + "_mad", expr(s"$median(${field}_diff, array(0.5), 100)").over(byColumns)(0))
          .drop(field + "_diff", field + "_median")
    }
    result
  }
}

object MAD extends ApplyModel {
  override def apply(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String], targetName: Option[String], keywords: Map[String, String], utils: PluginUtils): DataFrame => DataFrame =
    MAD(featureCols, keywords, searchId, utils).makePrediction
}
