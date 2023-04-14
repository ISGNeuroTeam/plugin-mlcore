package com.isgneuro.otp.plugins.mlcore.classification

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

case class DecisionTreeClf(featureCols: List[String], targetOption: Option[String], modelName: String,
                                  keywords: Map[String, String], searchId: Int, utils: PluginUtils) {
  import utils._

  val targetCol: String = targetOption.getOrElse(sendError(searchId, "Target column should be defined"))

  val MAX_DEPTH_DEFAULT = 5
  val HANDLE_NA_DEFAULT = "drop"

  def createPipeline(df: DataFrame): Pipeline = {
    val max_depth = Caster.safeCast[Int](
      keywords.get("max_depth"),
      MAX_DEPTH_DEFAULT,
      sendError(searchId, "The value of parameter 'max_depth' should be of boolean type")
    )

    val featuresName = s"__${modelName}_features__"
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol(featuresName)

    val indexedLabel = s"__indexed${targetCol}__"
    val labelIndexer = new StringIndexer()
      .setInputCol(targetCol)
      .setOutputCol(indexedLabel)
      .setStringOrderType("alphabetAsc")
      .fit(df)

    val dt = new DecisionTreeClassifier()
      .setLabelCol(indexedLabel)
      .setFeaturesCol(featuresName)
      .setPredictionCol(modelName + "_index_prediction")
      .setMaxDepth(max_depth)

    val labelConverter = new IndexToString()
      .setInputCol(modelName + "_index_prediction")
      .setOutputCol(modelName + "_prediction")
      .setLabels(labelIndexer.labels)

    new Pipeline().setStages(Array(assembler, labelIndexer, dt, labelConverter))
  }

  def prepareDf(dataFrame: DataFrame): DataFrame = {
    val handleNA = Caster.safeCast[String](
      keywords.get("na"),
      HANDLE_NA_DEFAULT,
      sendError(searchId, "Incorrect value of parameter 'na'")
    )

    dataFrame
      .drop(modelName + "_prediction")
      .drop("probability")
      .filter(targetCol + " is not null")
      .transform(
        handleNA match {
          case "drop" => (_df: DataFrame) => _df.na.drop("any", featureCols)
          case "keep" => (_df: DataFrame) => _df
          case _ => (_df: DataFrame) => _df
        }
      )
  }

  def makePrediction(df: DataFrame): (PipelineModel, DataFrame) = {
    val vecToArray = udf((xs: linalg.Vector) => xs.toArray)
    val pipeline = createPipeline(df)
    val prepared_df = prepareDf(df)
    val pModel = pipeline.fit(prepared_df)
    val transformed_df = pModel
      .transform(df)
    val return_df = transformed_df
      .drop(s"__${modelName}_features__")
      .withColumn("probability", vecToArray(transformed_df("probability")))
    (pModel, return_df
      .drop(return_df.columns.filter(_.matches(".*index.*")): _*)
      .drop("rawPrediction")
    )
  }
}

object DecisionTreeClf extends FitModel {
  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      val result: (PipelineModel, DataFrame) =
        DecisionTreeClf(featureCols, targetCol, modelName, keywords, searchId, utils).makePrediction(df)
      result
    }
}
