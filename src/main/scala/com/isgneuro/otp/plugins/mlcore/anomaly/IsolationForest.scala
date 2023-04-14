package com.isgneuro.otp.plugins.mlcore.anomaly

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

case class IsolationForest(featureCols: List[String], modelName: String,
                           keywords: Map[String, String], searchId: Int, utils: PluginUtils) {

  val NUM_ESTIMATORS_DEFAULT = 100
  val CONTAMINATION_DEFAULT = 0.1
  val MAX_SAMPLES_DEFAULT = 1.0
  val BOOTSTRAP_DEFAULT = false
  val MAX_FEATURES_DEFAULT = 1.0

  /**
   *
   * @return input dataframe with added columns for outlierScore and anomaly binary prediction.
   *
   *         keywords:
   *  - num_estimators: The number of base estimators in the ensemble. Default: 100.
   *  - max_samples: The number of samples to draw from X to train each base estimator. Default: 1.0.
   *  - max_features: The number of features to draw from X to train each base estimator. Default: 1.0.
   *  - contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Default: 0.1.
   *  - bootstrap: If True, individual trees are fit on random subsets of the training data sampled with replacement. Default: false.
   */

  import utils._

  def createPipeline(): Pipeline = {

    val numEstimators = Caster.safeCast[Int](
      keywords.get("num_estimators"),
      NUM_ESTIMATORS_DEFAULT,
      sendError(searchId, "The value of parameter 'num_estimators' should be of int type")
    )
    val maxSamples = Caster.safeCast[Double](
      keywords.get("max_samples"),
      MAX_SAMPLES_DEFAULT,
      sendError(searchId, "The value of parameter 'max_samples' should be of double type")
    )
    val maxFeatures = Caster.safeCast[Double](
      keywords.get("max_features"),
      MAX_FEATURES_DEFAULT,
      sendError(searchId, "The value of parameter 'max_features' should be of double type")
    )
    val contamination = Caster.safeCast[Double](
      keywords.get("contamination"),
      CONTAMINATION_DEFAULT,
      sendError(searchId, "The value of parameter 'contamination' should be of double type")
    )
    val bootstrap = Caster.safeCast[Boolean](
      keywords.get("bootstrap"),
      BOOTSTRAP_DEFAULT,
      sendError(searchId, "The value of parameter 'bootstrap' should be of boolean type")
    )

    val featuresName = s"__${modelName}_features__"
    val featureAssembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol(featuresName)

    val isolationForest = new com.linkedin.relevance.isolationforest.IsolationForest()
      .setNumEstimators(numEstimators)
      .setBootstrap(bootstrap)
      .setMaxSamples(maxSamples)
      .setMaxFeatures(maxFeatures)
      .setFeaturesCol(featuresName)
      .setPredictionCol("anomaly")
      .setScoreCol("outlierScore")
      .setContamination(contamination)
      .setContaminationError(0.01 * contamination)
      .setRandomSeed(1)

    new Pipeline().setStages(Array(featureAssembler, isolationForest))
  }

  def makePrediction(dataFrame: DataFrame): (PipelineModel, DataFrame) = {
    val p = createPipeline()
    val pModel = p.fit(dataFrame.drop("anomaly", "outlierScore"))
    val rdf = pModel.transform(dataFrame)
    (pModel, rdf.drop(s"__${modelName}_features__"))
  }
}

object IsolationForest extends FitModel {
  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      val result: (PipelineModel, DataFrame) = IsolationForest(featureCols, modelName, keywords, searchId, utils).makePrediction(df)
      result
    }
}