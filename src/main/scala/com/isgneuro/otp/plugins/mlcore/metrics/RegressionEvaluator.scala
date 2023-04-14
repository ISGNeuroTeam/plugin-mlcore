package com.isgneuro.otp.plugins.mlcore.metrics

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import ot.dispatcher.plugins.small.sdk.ScoreModel
import ot.dispatcher.sdk.PluginUtils

object RegressionEvaluator extends ScoreModel {

  val SUPPORTED_METRICS = List("mse", "rmse", "mae", "mape", "smape", "r2", "r2_adj", "hist")
  val METRIC_DEFAULT = "rmse"
  val FEATURES_DEFAULT = 1
  val BUCKETS_DEFAULT = 10

  /** Metrics for regression
   */

  override def score(modelName: String,
                     modelConfig: Option[Config],
                     searchId: Int,
                     labelCol: String,
                     predictionCol: String,
                     keywords: Map[String, String],
                     utils: PluginUtils):
  DataFrame => DataFrame =
    df => {
      import utils._
//      val log = getLoggerFor(this.getClass.getName)

      val metricName = keywords.get("metric") match {
        case Some(m) if SUPPORTED_METRICS.contains(m) => m
        case Some(_) => utils.sendError(searchId, "No such metric. Available metrics: \"mse\", \"rmse\", \"mae\", \"mape\", \"smape\", \"r2\", \"r2_adj\", \"hist\"")
        case None => METRIC_DEFAULT
      }

      val featuresNumber = Caster.safeCast[Int](
        keywords.get("features"),
        FEATURES_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'features' should be of int type")
      )

      val bucketsNumber = Caster.safeCast[Int](
        keywords.get("buckets"),
        BUCKETS_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'buckets' should be of int type")
      )

      def makeEvaluate(df: DataFrame): DataFrame = {
        val resultDF = metricName match {
          case "mse" => calcMse(df)
          case "rmse" => calcRmse(df)
          case "mae" => calcMae(df)
          case "mape" => calcMape(df)
          case "smape" => calcSmape(df)
          case "r2" => calcR2(df)
          case "r2_adj" => calcAdjR2(df, featuresNumber)
          case "hist" => calcHist(df, bucketsNumber)
        }
        resultDF
      }
   
      /**
       * Returns the mean squared error
       *
       */ 
      def calcMse(df: DataFrame): DataFrame = {
        val resultDf = df.withColumn("error", pow(df(labelCol) - df(predictionCol), 2))
          .select("error")
          .agg(avg("error"))
        resultDf
      }

      /**
       * Returns the root mean squared error
       *
       */
      def calcRmse(df: DataFrame): DataFrame = {
        val resultDf = df.withColumn("error", pow(df(labelCol) - df(predictionCol), 2))
          .select("error")
          .agg(sqrt(avg("error")) as metricName)
        resultDf
      }

      /**
       * Returns the mean absolute error
       *
       */
      def calcMae(df: DataFrame): DataFrame = {
        val resultDF = df.withColumn("error", abs(df(labelCol) - df(predictionCol)))
          .select("error")
          .agg(avg("error") as metricName)
        resultDF
      }

      /**
       * Returns the mean absolute percentage error
       *
       */
      def calcMape(df: DataFrame): DataFrame = {
        val resultDF = df
          .withColumn("error", abs((df(labelCol) - df(predictionCol)) / df(labelCol)))
          .select("error")
          .agg(avg("error") as metricName)
        resultDF
      }

      /**
       * Returns the symmetric mean absolute percentage error
       *
       */
      def calcSmape(df: DataFrame): DataFrame = {

        val numenatorDenominatorDf = df
          .withColumn("numenator", abs(df(labelCol) - df(predictionCol)))
          .withColumn("denominator", (abs(df(labelCol)) + abs(df(predictionCol))) / 2)
        val meanFractDf = numenatorDenominatorDf
          .withColumn("fract", numenatorDenominatorDf("numenator") / numenatorDenominatorDf("denominator"))
          .select("fract")
          .agg(avg("fract") as "meanFract")
        val resultDf = meanFractDf
          .withColumn(metricName, meanFractDf("meanFract") * lit(100))
          .select(metricName)
        resultDf
      }

      /**
       * Returns the coefficient of determination (R2)
       *
       */
      def calcR2(df: DataFrame): DataFrame = {
        val meanDf = df
          .select(labelCol)
          .agg(avg(labelCol) as "meanLabelCol")
        val meanFullDf = df.crossJoin(meanDf)
        val dfWithSS = meanFullDf
          .withColumn("squareResidual", pow(meanFullDf(labelCol) - meanFullDf(predictionCol), 2))
          .withColumn("squareTotal", pow(meanFullDf(labelCol) - meanFullDf("meanLabelCol"), 2))
          .select("squareTotal", "squareResidual")
          .agg(sum("squareResidual") as "sumSquareResidual", sum( "squareTotal") as "sumSquareTotal")
        val resultDf = dfWithSS
          .withColumn(metricName, lit(1) - dfWithSS("sumSquareResidual") / dfWithSS("sumSquareTotal"))
          .select(metricName)
        resultDf
      }
 
      /**
       * Returns the adjusted coefficient of determination (R2_adj)
       *
       */
      def calcAdjR2(df: DataFrame, featuresNumber: Int): DataFrame = {
        val lengthDf = df.count()
        val coefDetermination = calcR2(df)
        val resultDf = coefDetermination
          .withColumn(metricName, lit(1) - (lit(1) - coefDetermination("r2_adj")) * lit(lengthDf - 1) / lit(lengthDf - featuresNumber))
          .select(metricName)
        resultDf
      }

      def calcHist(df: DataFrame, bucketsNumber: Int): DataFrame = {
        val sqc = utils.spark.sqlContext
        import sqc.implicits._
        val sc = utils.spark.sparkContext

        val residualsDf = df.withColumn("residuals", df(labelCol) - df(predictionCol))
          .select("residuals")
        val (boundaries, values) = residualsDf.rdd.map(value => value.getDouble(0)).histogram(bucketsNumber)

        val froms = boundaries.dropRight(1).map(x => "%.3f".format(x).toDouble)
        val tos = boundaries.drop(1).map(x => "%.3f".format(x).toDouble)

        val xs = Array(froms,tos,values.map(x => x.toDouble)).transpose
        val rdd = sc.parallelize(xs).map(ys => (ys(0), ys(1), ys(2)))
        val resultDf = rdd.toDF("from","to","count")

        resultDf
      }


      val newDf = makeEvaluate(df)
      newDf
    }
}
