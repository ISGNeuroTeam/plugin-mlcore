package com.isgneuro.otp.plugins.mlcore.ts

import com.typesafe.config.Config
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils
import breeze.stats.distributions.Gaussian
import com.isgneuro.otp.plugins.mlcore.util.{Caster, Datetime}
import org.apache.log4j.Logger


object ConversionUtils extends Serializable {
  def MannKendallTest(ts: Seq[Double], alpha: Double, eps: Double): Double = {
    if (ts.length < 2) return 0.0

    def sgn_with_eps(a: Double, eps: Double): Double = {
      a match {
        case x if x > eps => 1.0
        case x if scala.math.abs(x) <= eps => 0.0
        case x if x < -eps => -1.0
      }
    }

    val S = ts.combinations(2).map(x => sgn_with_eps(x(1) - x.head, eps)).sum
    val N = ts.length

    val unique_ts = ts.distinct.size
    val tp = ts.groupBy(l => l).map(t => t._2.length).map(x => x * (x - 1) * (2 * x + 5)).sum

    val var_s =
      if (unique_ts == N) (N * (N - 1) * (2 * N + 5)) / 18
      else (N * (N - 1) * (2 * N + 5) - tp) / 18

    val z = S match {
      case x if x > eps => (x - 1.0) / scala.math.sqrt(var_s)
      case x if x < -eps => (x + 1.0) / scala.math.sqrt(var_s)
      case _ => 0.0
    }

    val norm = new Gaussian(0, 1)
    val h = scala.math.abs(z) > norm.inverseCdf(1 - alpha / 2)
    val trend = if (h) scala.math.signum(z) else 0.0
    trend
  }
}

case class MannKendallTest(
    fieldsUsed: List[String],
    properties: Map[String, String],
    searchId: Int,
    utils: PluginUtils) {
  import utils._

  val log: Logger = getLoggerFor(this.getClass.getName)

  private val BY_DEFAULT = List[String]()
  private val WINDOW_DEFAULT = Tuple2[Long, Long](Window.unboundedPreceding, Window.unboundedFollowing)
  private val OVER_DEFAULT = "__idx__"
  private val DEFAULT_ALPHA = 0.05
  private val DEFAULT_EPS = 0.1

  def makePrediction(dataFrame: DataFrame): DataFrame = {

    val by = properties.get("by") match {
      case Some(m) => m.split(",").map(_.trim).toList
      case None => BY_DEFAULT
    }

    def get_window(m: String): Long = {
      m match {
        case x if x matches Datetime.pattern_timespan => Datetime.getSpanInSeconds(x)
        case x if x matches Datetime.pattern_int => x.toLong
        case _ =>
          sendError(
            searchId,
            "The value of parameter 'window' should be of integer type or string type with duration. " +
              "Example: 4h, 8min, 7w. See a documentation")
          0L
      }
    }
    val window = properties.get("window") match {
      case Some(m) => Tuple2[Long, Long](1 - get_window(m), Window.currentRow)
      case _ => WINDOW_DEFAULT
    }
    val order = properties.get("order") match {
      case Some(m) => m
      case _ => OVER_DEFAULT
    }
    
    def udf_MankdallTest =
      udf[Double, Seq[Double], Double, Double](ConversionUtils.MannKendallTest)

    val alpha = Caster.safeCast[Double](
      properties.get("alpha"),
      DEFAULT_ALPHA,
      utils.sendError(searchId, "The value of parameter 'alpha' should be of double type"))

    val eps = Caster.safeCast[Double](
      properties.get("eps"),
      DEFAULT_EPS,
      utils.sendError(searchId, "The value of parameter 'eps' should be of double type"))

    def get_windowType(m: String): String = {
      m match {
        case x if x matches Datetime.pattern_timespan => "time_window"
        case x if x matches Datetime.pattern_int => "int_window"
        case _ => "no type"
      }
    }

    val window_type  = properties.get("window") match {
      case Some(m) => get_windowType(m)
      case _ => "int_window"
    }

    val overWindow = window_type match {
      case "time_window" => Window.orderBy(order).partitionBy(by.map(col): _*).rangeBetween(window._1, window._2)
      case "int_window" => Window.orderBy(order).partitionBy(by.map(col): _*).rowsBetween(window._1, window._2)
      case _ => utils.sendError(searchId, s"Can't create windowSpec with type $window_type ")
    }
    val result = dataFrame
      .withColumn(OVER_DEFAULT, monotonically_increasing_id)
      .withColumn(
        "trend",
        udf_MankdallTest(collect_list(col(fieldsUsed.head)).over(overWindow), lit(alpha), lit(eps)))
    result
  }
}

object MannKendallTest extends ApplyModel {
  override def apply(
      modelName: String,
      modelConfig: Option[Config],
      searchId: Int,
      featureCols: List[String],
      targetName: Option[String],
      keywords: Map[String, String],
      utils: PluginUtils): DataFrame => DataFrame =
    MannKendallTest(featureCols, keywords, searchId, utils).makePrediction
}
