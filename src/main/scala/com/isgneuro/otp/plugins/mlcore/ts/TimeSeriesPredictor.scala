package com.isgneuro.otp.plugins.mlcore.ts

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{max, _}
import org.apache.spark.sql.types.{DoubleType, LongType}
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils
import ot.dispatcher.sdk.core.extensions.DataFrameExt._
import ot.dispatcher.sdk.core.functions.Datetime

import scala.util.{Failure, Success, Try}

object TimeSeriesPredictor extends ApplyModel {

  val FUTURE_DEFAULT = 100
  val MODEL_TYPE_DEFAULT = "additive"
  val MODE_DEFAULT = "single-trend"
  val PERIOD_DEFAULT = "1d"
  val SUPPORTED_MODELS = List("additive", "multiplicative")
  val SUPPORTED_TREND_MODES = List("single-trend", "multi-trend")


  /** Time series prediction algorithm.
   * works in two modes: predicting with a linear trend and with a polygonal trend (trend with changepoints). Overall idea:
   * without trend changepoints:
   * 1. remove trend, either additively or multiplicatively
   * 2. generate features from time series
   * 3. fit linear regression or detrended series
   * 4. predict for current dataframe
   * 5. generate future dataframe
   * 6. predict for future dataframe
   * 7. append current dataframe and future dataframe
   * 8. restore trend
   *
   * with trend changepoints:
   * 1. detect trend changepoints
   * 2. split current dataframe dy trend changepoints
   * 3. detect trends coefficients
   * 4. remove trends
   * 5. merge all detrended parts into one
   * 6. fit linear regression or detrended series
   * 7. predict for current dataframe
   * 8. generate future dataframe
   * 9. predict for future dataframe
   * 10. restore trends for current dataframe
   * 11. use last trend coefficients to restore trend for future dataframe
   * 12. append current and future dataframes
   *
   * Note: to use saved model, all feature columns (month, dayofweek, etc) should be created before calling apply method
   */

  override def apply(modelName: String,
                   modelConfig: Option[Config],
                   searchId: Int,
                   featureCols: List[String],
                   targetCol: Option[String],
                   keywords: Map[String, String],
                   utils: PluginUtils):
  DataFrame => DataFrame =
    df => {

      import utils._
      val log = getLoggerFor(this.getClass.getName())

      val future = Caster.safeCast[Int](
        keywords.get("future"),
        FUTURE_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'future' should be of int type")
      )

      val modelType = keywords.get("modelType") match {
        case Some(m) if SUPPORTED_MODELS.contains(m) => m
        case Some(_) => utils.sendError(searchId, "No such strategy. Available strategies: auto, all, onethird, sqrt, log2")
        case None => MODEL_TYPE_DEFAULT
      }
      val mode = keywords.get("mode") match {
        case Some(m) if SUPPORTED_TREND_MODES.contains(m) => m
        case Some(_) => utils.sendError(searchId, "No such strategy. Available modes: single-trend, multi-trend")
        case None => MODE_DEFAULT
      }
      val timeColumn = keywords.getOrElse("timeColumn", "_time")
      val targetColumn = Try(targetCol.get) match{
        case Success(n) => n
        case Failure(_) => utils.sendError(searchId, "The name of a target column was not set")
      }

      val period = keywords.get("period") match {
        case Some(m) => Try(Datetime.getSpanInSeconds(m)) match {
          case Success(n) => n
          case Failure(_) => utils.sendError(searchId, "Could not parse period")
        }
        case None => Datetime.getSpanInSeconds(PERIOD_DEFAULT)
      }

      def featureCreator(_df: DataFrame): DataFrame = {
        val featureDf = _df.withColumn("time_stamp", from_unixtime(col(s"$timeColumn"), "yyyy-MM-dd HH:mm"))
          .withColumn("month", month(col("time_stamp")))
          .withColumn("dayofmonth", dayofmonth(col("time_stamp")))
          .withColumn("dayofweek", dayofweek(col("time_stamp")) cast DoubleType)
          .withColumn("dayofyear", dayofyear(col("time_stamp")) cast DoubleType)
          .withColumn("minute", minute(col("time_stamp")))
          .withColumn("second", second(col("time_stamp")))
          .withColumn("year", year(col("time_stamp")))
          .withColumn("hour", hour(col("time_stamp")))
          .withColumn("week", weekofyear(col("time_stamp")))
          .withColumn("is_weekend", when(col("dayofweek") === 1, 1)
            .when(col("dayofweek") === 7, 1)
            .otherwise(0))
        featureDf
      }

      def trendCalculator(dataFrame: DataFrame): (Double, Double) = {
        val trendAssembler = new VectorAssembler()
          .setInputCols(Array(s"$timeColumn"))
          .setOutputCol("features")
        val trendpred = trendAssembler.transform(dataFrame)
          .withColumnRenamed(targetColumn, "label")
          .withColumn("label", col("label") cast DoubleType)
        val lrTrend = new LinearRegression().setMaxIter(10).setRegParam(0.1).setElasticNetParam(1)
        val lrTrendModel = lrTrend.fit(trendpred)
        val a = lrTrendModel.coefficients.toArray.head
        val b = lrTrendModel.intercept.toDouble
        log.debug(s"Linear trend coefficients: a =$a, intercept (b) = $b")
        (a, b)
      }

      def trendRemover(dataFrame: DataFrame, a: Double, b: Double, modelType: String): DataFrame = {
        if (modelType == "additive") {dataFrame.withColumn(targetColumn, col(targetColumn) - ((col(s"$timeColumn") * a) + b))}
        else {dataFrame.withColumn(targetColumn, col(targetColumn) / ((col(s"$timeColumn") * a) + b))}
      }

      def trendRestorer(dataFrame: DataFrame, a: Double, b: Double, modelType: String, column: String): DataFrame = {
        if (modelType == "additive") {dataFrame.withColumn(s"$column", col(s"$column") + ((col(s"$timeColumn") * a) + b))}
        else {dataFrame.withColumn(s"$column", col(s"$column") * ((col(s"$timeColumn") * a) + b))}
      }

      //function used in changepointDetector
      def cost_function(signal: Array[Double], start: Int, end: Int): Double = {
        val sub: Array[Double] = signal.slice(from = start, until = end + 1)
        val mean: Double = sub.sum / sub.length
        sub.map(x => math.pow(x - mean, 2)).sum
      }

      // PELT algorithm: http://article.sciencepublishinggroup.com/html/10.11648.j.ajtas.20150406.30.html , https://aakinshin.net/posts/edpelt/
      def changepointDetector(_df: DataFrame): Array[Int] = {
        val sample = _df.select(s"$targetColumn").rdd.map(r => r(0)).collect.map(_.toString.toDouble)
        val n_samples: Int = sample.length
        val changePoints = new Array[List[Int]](n_samples + 1)
        changePoints(0) = Nil
        val penalty: Double = 10000
        val function_value = new Array[Double](n_samples + 1)
        function_value(0) = -penalty
        val admissible = (1 until n_samples).foldLeft(List(0)) {
          case(acc, tau_star) =>
            val f_array = (for (tau <- acc) yield (function_value(tau) + cost_function(sample, tau + 1, tau_star) + penalty, tau)).par
            function_value(tau_star) = f_array.min._1
            val tau_one = f_array.minBy(_._1)._2
            changePoints(tau_star) = ((tau_one + 1) :: changePoints(tau_one).reverse).reverse
            val rev = (tau_star :: acc.reverse).reverse
            for (t <- rev if function_value(t) + cost_function(sample, t + 1, tau_star) <= function_value(tau_star)) yield t
        }
        val bestPoints = (sample.length :: changePoints(n_samples - 1).tail.reverse).reverse
        val bestPoints0 = 0 +: bestPoints
        val x = (bestPoints zip bestPoints0).map({ case (a, b) => a - b }).toArray
        x
      }

      def dfSplitter(dataFrame: DataFrame, lengths: Array[Int]): Array[DataFrame] = {
        var rest = dataFrame
        val separated = Array[DataFrame]()
        lengths.foldLeft(separated) {
          (a, b) =>
            val part = rest.sort(s"$timeColumn").limit(b)
            rest = rest.except(part)
            a :+ part
        }
      }

      def dfMerger(dfs: Array[DataFrame]): DataFrame = {
        val df2 = spark.emptyDataFrame
        dfs.foldLeft(df2) {(a, b) => b.append(a)}
      }

      def trendLoopRemover(severalDataframes: Array[DataFrame], model: String): (Array[DataFrame], Array[(Double, Double)]) = {
        val separated = Array[DataFrame]()
        val coeffs = Array[(Double, Double)]()
        severalDataframes.foldLeft((separated, coeffs)) {
          (a, b) =>
            val (coef_a, coef_b) =  trendCalculator(b)
            val singleFrame = trendRemover(b, coef_a, coef_b, model)
            (a._1 :+ singleFrame, a._2 :+ (coef_a, coef_b))
        }
      }

      def trendLoopRestorer(detrended: DataFrame, lengths: Array[Int], coeffs: Array[(Double, Double)], model: String): DataFrame = {
        val restored = spark.emptyDataFrame
        var rest = detrended
        lengths.zip(coeffs).foldLeft(restored) {
          (a, b) =>
            var part = rest.sort(s"$timeColumn").limit(b._1)
            rest = rest.except(part)
            part = trendRestorer(part, b._2._1, b._2._2, model, "prediction")
            a.append(part)
        }
      }

      def outlierRemover(df: DataFrame): DataFrame = {
        val quantiles1 = df.stat.approxQuantile(s"$targetColumn", Array(0.15, 0.85), 0.0)
        val Q11 = quantiles1(0)
        val Q31 = quantiles1(1)
        val IQR1 = Q31 - Q11
        val lowerRange1 = Q11 - 1.5 * IQR1
        val upperRange1 = Q31 + 1.5 * IQR1
        val result = df.withColumn(targetColumn, when(col(s"$targetColumn") < lowerRange1, lowerRange1).when(col(s"$targetColumn") > upperRange1, upperRange1).otherwise(col(s"$targetColumn")))
        result
      }

      df.createOrReplaceTempView("_df")
      val df1 = df.drop("prediction")
      var train = df1.withColumn(s"$timeColumn", col(s"$timeColumn") cast LongType)
        .drop("_raw")
        .withColumn(s"$targetColumn", col(s"$targetColumn") cast DoubleType)
        .withColumn("time_stamp", from_unixtime(col(s"$timeColumn"), "yyyy-MM-dd HH:mm"))

      var changetest = Array(0)
      var coeffs = Array(Tuple2(0.0, 0.0))
      var trendCoeffs = Tuple2(0.0, 0.0)
      if (mode == "multi-trend") {
        changetest = changepointDetector(df1)
        val trains = dfSplitter(train, changetest)
        val detrended = trendLoopRemover(trains, modelType)
        val detrendedDfs = detrended._1
        coeffs = detrended._2
        train = dfMerger(detrendedDfs)
      }
      else {
        trendCoeffs = trendCalculator(train)
        train = trendRemover(train, trendCoeffs._1, trendCoeffs._2, modelType)
      }

      train = outlierRemover(train)
      train = featureCreator(train)

      //linreg
      val assembler = new VectorAssembler()
        .setInputCols(Array("month", "dayofweek",
          "year", "is_weekend",
          "week", "dayofyear", "dayofmonth",
          "hour", "minute", "second"
        ))
        .setOutputCol("features").setHandleInvalid("skip")
      train = assembler.transform(train)
      train = train.withColumnRenamed(targetColumn, "label")
        .withColumn("label", col("label") cast DoubleType)
      val lr = new LinearRegression()
        .setMaxIter(30)
        .setRegParam(0.01)
        .setRegParam(0.01)
        .setElasticNetParam(1)
      val lrModel = lr.fit(train)
      log.debug(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

      // creating dataframe for future predictions
      val maxTime = train.agg(max(s"$timeColumn") cast LongType).collect()(0).getLong(0)
      var topredict = spark.range(maxTime + period, maxTime + period * (future + 1), period)
        .withColumnRenamed("id", s"$timeColumn")
        .withColumn("time_stamp", from_unixtime(col(s"$timeColumn")))
      topredict = featureCreator(topredict)

      topredict = assembler.transform(topredict)
      train = train.drop("label")
      var futurelrPrediction = lrModel.transform(topredict)
      futurelrPrediction = futurelrPrediction.select(s"$timeColumn", "prediction")

      var currentDf = df1.join(lrModel.transform(train).select(s"$timeColumn", "prediction"), usingColumns = Seq(s"$timeColumn"), joinType = "left").distinct()

      //restoring trend separately on current and future dfs

      if (mode == "multi-trend") {
        currentDf = trendLoopRestorer(currentDf, changetest, coeffs, modelType)
        futurelrPrediction = trendRestorer(futurelrPrediction, coeffs.last._1, coeffs.last._2, modelType, "prediction")
      }
      else {
        currentDf = trendRestorer(currentDf, trendCoeffs._1, trendCoeffs._2, modelType, "prediction")
        futurelrPrediction = trendRestorer(futurelrPrediction, trendCoeffs._1, trendCoeffs._2, modelType, "prediction")
      }

      //appending current and future
      currentDf = currentDf.append(futurelrPrediction)
      currentDf.sort("_time")
    }
}
