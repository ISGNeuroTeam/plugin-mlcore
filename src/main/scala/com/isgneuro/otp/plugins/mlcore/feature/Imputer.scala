package com.isgneuro.otp.plugins.mlcore.feature

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

case class Imputer(fieldsUsed: List[String], modelName: String, keywords: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val MISSING_VALUE_DEFAULT: Double = Double.NaN
  val STRATEGY_DEFAULT = "mean"
  val FILL_VALUE_DEFAULT = 0.0
  val FILL_VALUE_STRING_DEFAULT = "missing"
  val SUPPORTED_STRATEGIES = List("median", "mean")
  val DROP_STRATEGY_DEFAULT = "any"
  val SUPPORTED_DROP_STRATEGIES = List("any", "all")
  val NEW_COLUMNS_DEFAULT: Boolean = false

  /**
   *
   * @return input dataframe with added pca columns
   *
   *         keywords:
   *  - missing_value: The placeholder for the missing values. All occurrences of missingValue will be imputed.
   *    Note that null values are always treated as missing. Default: Double.NaN
   *  - strategy: The imputation strategy. Defaul: mean.
   *  - new_columns: If true, output columns will be added with names of input columns + "_imputed";
   *    if false, filling is done on input columns. Default: false.
   */

  def makePrediction(inputDf: DataFrame): (PipelineModel, DataFrame) = {
    val missingValue = Caster.safeCast[Double](
      keywords.get("missing_value"),
      MISSING_VALUE_DEFAULT,
      sendError(searchId, "The value of parameter 'missing_value' should be of double type")
    )
    val strategy = keywords.get("strategy") match {
      case Some(m) if SUPPORTED_STRATEGIES.contains(m) => m
      case Some(_) => utils.sendError(searchId, s"""No such strategy. Available strategies: ${SUPPORTED_STRATEGIES.mkString(", ")}.""")
      case None => STRATEGY_DEFAULT
    }
    val newColumns = Caster.safeCast[Boolean](
      keywords.get("new_columns"),
      NEW_COLUMNS_DEFAULT,
      sendError(searchId, "The value of parameter 'new_columns' should be of boolean type")
    )

    val outputColumns = fieldsUsed.toArray.map(_ + "_imputed")
    val dataFrame = inputDf.drop(outputColumns: _*)

    val imputer = new org.apache.spark.ml.feature.Imputer()
      .setInputCols(fieldsUsed.toArray)
      .setMissingValue(missingValue)
      .setOutputCols(outputColumns)
      .setStrategy(strategy)

    val pipeline = new Pipeline().setStages(Array(imputer))
    val pipelineModel = pipeline.fit(dataFrame)
    val imputedDf = pipelineModel.transform(dataFrame)
    val outDf = if (!newColumns) {
      val droppedDf = imputedDf.drop(fieldsUsed: _*)
      val renamedCols = fieldsUsed.zip(outputColumns)
      renamedCols.foldLeft(droppedDf)((acc, c) => acc.withColumnRenamed(c._2, c._1))
    }
    else imputedDf
    (pipelineModel, outDf)
  }
}

object Imputer extends FitModel {

  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      val result: (PipelineModel, DataFrame) = Imputer(featureCols, modelName, keywords, searchId, utils).makePrediction(df)
      result
    }
}
