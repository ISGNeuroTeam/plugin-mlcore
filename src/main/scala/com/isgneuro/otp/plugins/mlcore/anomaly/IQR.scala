package com.isgneuro.otp.plugins.mlcore.anomaly

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame}
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils

case class IQR(fieldsUsed: List[String], properties: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val RANGE_DEFAULT: List[Double] = List[Double](0.25, 0.75)
  val WITH_CENTERING_DEFAULT = true
  val WITH_SCALING_DEFAULT = true
  val BY_DEFAULT: List[String] = List[String]()
  val WINDOW_BEFORE_DEFAULT: Long = -1 * Window.unboundedPreceding
  val WINDOW_AFTER_DEFAULT: Long = Window.unboundedFollowing

  /**
   *
   * @return input dataframe with added column for each input feature with computed Interquartile Range
   *
   *         keywords:
   *  - range: quartiles to compute IQR. Default: (0.25,0.75)
   *  - with_centering: Defines if values should be centered. Default: true.
   *  - with_scaling: Defines if values should be scaled. Default: true.
   */

  def makePrediction(dataFrame: DataFrame): DataFrame = {

    val List(q1, q3) = properties.get("range") match {
      case Some(m) => m.filterNot(c => c == '(' || c == ')' || c == ' ').split(",").map(_.toDouble).toList
      case None => RANGE_DEFAULT
    }
    val withCentering = Caster.safeCast[Boolean](
      properties.get("with_centering"),
      WITH_CENTERING_DEFAULT,
      sendError(searchId, "The value of parameter 'with_centering' should be of boolean type")
    )
    val withScaling = Caster.safeCast[Boolean](
      properties.get("with_scaling"),
      WITH_SCALING_DEFAULT,
      sendError(searchId, "The value of parameter 'with_scaling' should be of boolean type")
    )
    val by = properties.get("by") match {
      case Some(m) => m.split(",").map(_.trim).toList
      case None => BY_DEFAULT
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

    def get_iqr(q1: Column, q2: Column, q3: Column)(x: Column): Column = {
      val resultIfCentering = if (withCentering) x - q2 else x
      if (withScaling) resultIfCentering / (q3 - q1) else resultIfCentering
    }

    val byColumns = Window.partitionBy(by.map(col): _*).rowsBetween(-1 * window_before, window_after)

    val result = fieldsUsed.foldLeft(dataFrame) {
      case (acc, field) =>
        val withQ1 = acc.withColumn(field + "_q1", expr(s"approx_percentile($field, array($q1))").over(byColumns)(0))
        val withQ2 = withQ1.withColumn(field + "_q2", expr(s"approx_percentile($field, array(0.5))").over(byColumns)(0))
        val withQ3 = withQ2.withColumn(field + "_q3", expr(s"approx_percentile($field, array($q3))").over(byColumns)(0))
        val withIQR = withQ3.withColumn("iqr_" + field, get_iqr(col(field + "_q1"), col(field + "_q2"), col(field + "_q3"))(col(field)))
        withIQR.drop(field + "_q1", field + "_q2", field + "_q3")
    }
    result
  }
}

object IQR extends ApplyModel {
  override def apply(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String], targetName: Option[String], keywords: Map[String, String], utils: PluginUtils): DataFrame => DataFrame =
    IQR(featureCols, keywords, searchId, utils).makePrediction
}
