package com.isgneuro.otp.plugins.mlcore.text

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

case class TFIDF(featureCol: String, modelName: String, keywords: Map[String, String], searchId: Int, utils: PluginUtils) {

  val DEFAULT_NUM_FEATURES = 20
  private val numFeatures = Caster.safeCast[Int](
    keywords.get("num_features"),
    DEFAULT_NUM_FEATURES,
    utils.sendError(searchId, "Wrong number of features"))

  def transform(df: DataFrame): (PipelineModel, DataFrame) = {
    val tokenizer = new Tokenizer()
      .setInputCol(featureCol)
      .setOutputCol("__words__")

    val hashingTF = new HashingTF()
      .setInputCol("__words__")
      .setOutputCol("rawFeatures")
      .setNumFeatures(numFeatures)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf))
    val pipelineModel = pipeline.fit(df)
    val result = pipelineModel.transform(df).drop("__words__", "rawFeatures")
    (pipelineModel, result)
  }
}

object TFIDF extends FitModel {

  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      featureCols match {
        case featureCol :: _ => TFIDF(featureCol, modelName, keywords, searchId, utils).transform(df)
        case _ => utils.sendError(searchId, "No feature column specified")
      }
    }
}
