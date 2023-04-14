package com.isgneuro.otp.plugins.mlcore.text

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils
import org.apache.spark.ml.feature.{CountVectorizer, StopWordsRemover, Tokenizer}

case class SmallCountVectorizer(featureCol: String, modelName: String, keywords: Map[String, String], searchId: Int, utils: PluginUtils) {
  private val minWordCount = Caster.safeCast[Int](
    keywords.get("min_count"),
    3,
    utils.sendError(searchId, "Cannot cast min_count to Integer")
  )

  private val removeStopWords = Caster.safeCast[Boolean](
    keywords.get("stop_words"),
    true,
    utils.sendError(searchId, "Cannot cast stop_words to Boolean")
  )

  def transform(df: DataFrame): (PipelineModel, DataFrame) = {
    val tokenizer = new Tokenizer()
      .setInputCol(featureCol)
      .setOutputCol("__words__")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("__words__")
      .setOutputCol("__remove_stop__")

    val cv = new CountVectorizer()
      .setInputCol("__remove_stop__")
      .setOutputCol("features")
      .setMinDF(minWordCount)
      .setVocabSize(1000)

    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, cv))
    val pipelineModel = pipeline.fit(df)
    val result = pipelineModel.transform(df).drop("__words__", "__remove_stop__")
    (pipelineModel, result)
  }
}

object SmallCountVectorizer extends FitModel {

  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      featureCols match {
        case featureCol :: _ => SmallCountVectorizer(featureCol, modelName, keywords, searchId, utils).transform(df)
        case _ => utils.sendError(searchId, "No feature column specified")
      }
    }
}