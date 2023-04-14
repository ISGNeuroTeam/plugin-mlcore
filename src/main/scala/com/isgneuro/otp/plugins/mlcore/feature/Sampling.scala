package com.isgneuro.otp.plugins.mlcore.feature

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{array_repeat, col, explode}
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils


case class Sampling(labelOption: Option[String], keywords: Map[String, String], id: Int, utils: PluginUtils) {

  import utils._

  val labelCol: String = labelOption.getOrElse(sendError(id, "Target column should be defined"))

  val WITH_REPLACEMENT_DEFAULT = false
  val SEED_DEFAULT: Long = scala.util.Random.nextLong()
  val SUPPORTED_METHOD = List("downsampling", "oversampling")

  /**
   *
   * @return sample from input dataframe by target column
   *
   *         keywords:
   *  - fraction: if fraction in range [0;1] - it's the fraction of majority class to be taken.
   *    If fraction >1.0, it's the ratio of minority class samples for oversampling. Default: balanced downsampling.
   *  - with_replacement: if true, sampling is done with replacement. Default: false.
   *  - seed: the sampling seed. Default: scala.util.Random.nextLong()
   *  - method: the sampling method, over- or downsampling. Default: downsampling.
   */

  def makePrediction(df: DataFrame): DataFrame = {

    val groupedData = df.groupBy(labelCol).count()
    require(groupedData.count == 2, "Only 2 labels allowed")
    val classAll = groupedData.sort(labelCol).collect()
    val minorityClass = classAll(1)(0).toString
    val minorityClassCount = classAll(1)(1).toString.toDouble
    val majorityClass = classAll(0)(0).toString
    val majorityClassCount = classAll(0)(1).toString.toDouble

    val inputFraction = Caster.safeCast[Double](
      keywords.get("fraction"),
      minorityClassCount / majorityClassCount,
      sendError(id, "The value of parameter 'fraction' should be of double type")
    ) match {
      case x if x >= 0 => x
      case _ => sendError(id, "The value of parameter 'fraction' should be nonnegative")
    }
    val withReplacement = Caster.safeCast[Boolean](
      keywords.get("with_replacement"),
      WITH_REPLACEMENT_DEFAULT,
      sendError(id, "The value of parameter 'with_replacement' should be of boolean type")
    )
    val seed = Caster.safeCast[Long](
      keywords.get("seed"),
      SEED_DEFAULT,
      sendError(id, "The value of parameter 'seed' should be of long type")
    )
    val method = keywords.get("method") match {
      case Some(m) => if (SUPPORTED_METHOD.contains(m)) m
      else sendError(id, "The value of parameter 'method' should be 'oversampling' or 'downsampling'")
      case None => if (inputFraction <= 1.0) "downsampling" else "oversampling"
    }

    val fraction =
      if (method == "oversampling" & inputFraction < 1.0 | method == "downsampling" & inputFraction > 1.0)
        1.0 / inputFraction
      else
        inputFraction

    // downsampling for majority class
    val majority = df.filter(col(labelCol) === majorityClass)
    val minority = df.filter(col(labelCol) === minorityClass)

    val oversampled_df = minority.withColumn(
      "dummy",
      explode(
        array_repeat(
          col(labelCol), math.floor(fraction).toInt))
    ).drop("dummy")

    val out = method match {
      case "downsampling" => majority.sample(withReplacement, fraction, seed).union(minority)
      case "oversampling" => majority.union(oversampled_df).union(minority.sample(fraction - math.floor(fraction), seed))
    }
    out
  }
}

object Sampling extends ApplyModel {
  override def apply(modelName: String,
                     modelConfig: Option[Config],
                     searchId: Int,
                     featureCols: List[String],
                     targetName: Option[String],
                     keywords: Map[String, String],
                     utils: PluginUtils): DataFrame => DataFrame =
    Sampling(targetName, keywords, searchId, utils).makePrediction
}


