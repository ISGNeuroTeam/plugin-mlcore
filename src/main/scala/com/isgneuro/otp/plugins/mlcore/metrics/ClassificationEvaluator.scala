package com.isgneuro.otp.plugins.mlcore.metrics

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.ScoreModel
import ot.dispatcher.sdk.PluginUtils
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object ClassificationEvaluator extends ScoreModel {

  val SUPPORTED_METRICS = List("accuracy", "precision", "recall", "f1", "roc_auc")
  val METRIC_DEFAULT = "accuracy"
  val WEIGHT_DEFAULT = 0.5


  /** Metrics for classification
   */

  override def score(modelName: String,
                     modelConfig: Option[Config],
                     searchId: Int,
                     labelCol: String,
                     predictionCol: String,
                     keywords: Map[String, String],
                     utils: PluginUtils):
  DataFrame => DataFrame = df => {
    import utils._
    val log = getLoggerFor(this.getClass.getName)


    val metricName = keywords.get("metric") match {
      case Some(m) if SUPPORTED_METRICS.contains(m) => m
      case Some(_) => utils.sendError(searchId, "No such metric. Available metrics: \"accuracy\", \"precision\", \"recall\", \"f1\", \"roc_auc\"")
      case None => METRIC_DEFAULT
    }

    val weight = Caster.safeCast[Double](
      keywords.get("weight"),
      WEIGHT_DEFAULT,
      utils.sendError(searchId, "The value of parameter 'weight' should be of double type")
    )

    def makeEvaluate(df: DataFrame): DataFrame = {
      val resultDF = metricName match {
        case "accuracy" => calcAccuracy(df)
        case "precision" => calcPrecision(df)
        case "recall" => calcRecall(df)
        case "f1" => calcFMeasure(df, weight)
        case "roc_auc" => calcRocAuc(df)
      }
      resultDF
    }

    /**
     * Returns the accuracy score
     */
    def calcAccuracy(df: DataFrame): DataFrame = {
      val resultDf = df
        .withColumn("matches", when(df(labelCol) === df(predictionCol), 1).otherwise(0))
        .select("matches")
        .agg(avg("matches") as metricName)
      resultDf
    }
    
    /**
     * Returns the precision score
     */
    def calcPrecision(df: DataFrame): DataFrame = {
      val trueFalsePositive = df
        .withColumn("truePositive", when(df(labelCol) === 1 and df(predictionCol) === 1, 1).otherwise(0))
        .withColumn("falsePositive", when(df(labelCol) === 0 and df(predictionCol) === 1, 1).otherwise(0))
        .select("truePositive", "falsePositive")
        .agg(sum("truePositive") as "truePositive", sum("falsePositive") as "falsePositive")
      val resultDf = trueFalsePositive
        .withColumn("precision", trueFalsePositive("truePositive") / (trueFalsePositive("truePositive") + trueFalsePositive("falsePositive")))
        .select("precision")
      resultDf
    }
 
    /**
     * Returns the recall score
     */
    def calcRecall(df: DataFrame): DataFrame = {
      val trueFalsePositive = df
        .withColumn("truePositive", when(df(labelCol) === 1 and df(predictionCol) === 1, 1).otherwise(0))
        .withColumn("falseNegative", when(df(labelCol) === 1 and df(predictionCol) === 0, 1).otherwise(0))
        .select("truePositive", "falseNegative")
        .agg(sum("truePositive") as "truePositive", sum("falseNegative") as "falseNegative")
      val resultDf = trueFalsePositive
        .withColumn("recall", trueFalsePositive("truePositive") / (trueFalsePositive("truePositive") + trueFalsePositive("falseNegative")))
        .select("recall")
      resultDf
    }

    /**
     * Returns the f-measure score
     */
    def calcFMeasure(df: DataFrame, weight: Double): DataFrame = {
      val precision = calcPrecision(df)
      val recall = calcRecall(df)
      val precisionRecall = precision.crossJoin(recall)
      val resultDf = precisionRecall
        .withColumn(metricName, lit(1 + weight * weight) * precisionRecall("precision") * precisionRecall("recall") / (lit(weight * weight) * precisionRecall("precision") + precisionRecall("recall")))
        .select(metricName)
      resultDf
    }

    /**
     * Returns the ROC-AUC score
     *
     */
    def calcRocAuc(df: DataFrame): DataFrame = {
      val truePredictDf = df
        .withColumn("truePredict", when(df(labelCol) === df(predictionCol), 1).otherwise(0))
      val n1 = truePredictDf
        .filter(truePredictDf("truePredict") === 1).count()
      val n0 = truePredictDf
        .filter(truePredictDf("truePredict") === 0).count()
      val rankedDf = truePredictDf
        .withColumn("rank", rank().over(Window.orderBy("truePredict")))
      val rankSumDf = rankedDf
        .filter(rankedDf(labelCol) === 1)
        .agg(sum("rank") as "rankSum")
      val u_df = rankSumDf
        .withColumn("U1", rankSumDf("rankSum") - lit(n1 * (n1 + 1) / 2))
        .select("U1")
      val resultDF = u_df
        .withColumn(metricName, lit(1) - u_df("U1") / lit(n1 * n0))
        .select(metricName)
      resultDF

    }

    makeEvaluate(df)
  }
}
