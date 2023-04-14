package com.isgneuro.otp.plugins.mlcore.classification

import com.isgneuro.otp.plugins.mlcore.util.{Caster, DoubleToIntTransformer}
import com.typesafe.config.Config
import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, XGBoostClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils
import org.apache.spark.sql.DataFrame

import scala.util.{Failure, Success, Try}


object XgboostClassifier extends FitModel {

  val LEARNING_RATE_DEFAULT = 0.1
  val RANDOM_SEED_DEFAULT = 42
  val NUM_TREES_DEFAULT = 100
  val TREE_METHOD_DEFAULT = "approx"
  val SUPPORTED_TREE_METHOD = List("approx", "hist")
  val MAX_DEPTH_DEFAULT = 4
  val LAMBDA_DEFAULT = 0 // Default value for L2 regularization
  val ALPHA_DEFAULT = 0 //  Default value for L1 regularization

  /**
   * Xgboost classifier algorithm.
   */

  override def fit(
      modelName: String,
      modelConfig: Option[Config],
      searchId: Int,
      featureCols: List[String],
      targetCol: Option[String],
      keywords: Map[String, String],
      utils: PluginUtils): DataFrame => (PipelineModel, DataFrame) =
    df => {

      def createPipeline() = {
        val targetColumn = Try(targetCol.get) match {
          case Success(n) => n
          case Failure(_) =>
            utils.sendError(searchId, "The name of a target column was not set")
        }

        val learningRate = Caster.safeCast[Double](
          keywords.get("learningRate"),
          LEARNING_RATE_DEFAULT,
          utils.sendError(
            searchId,
            "The value of parameter 'learningRate' should be of float type"))

        val seed = Caster.safeCast[Long](
          keywords.get("randomSeed"),
          RANDOM_SEED_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'seed' should be of float type"))

        val numTrees = Caster.safeCast[Int](
          keywords.get("numTrees"),
          NUM_TREES_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'numTrees' should be of int type"))

        val treeMethod = keywords.get("treeMethod") match {
          case Some(m) if SUPPORTED_TREE_METHOD.contains(m) => m
          case Some(_) =>
            utils.sendError(searchId, "No such strategy. Available strategies: approx hist")
          case None => TREE_METHOD_DEFAULT
        }

        val maxDepth = Caster.safeCast[Int](
          keywords.get("maxDepth"),
          NUM_TREES_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'maxDepth' should be of int type"))

        val alpha = Caster.safeCast[Double](
          keywords.get("alpha"),
          ALPHA_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'alpha' should be of double type"))
        val lambda = Caster.safeCast[Double](
          keywords.get("lambda"),
          LAMBDA_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'lambda' should be of double type"))

        val indexedLabel = s"__indexed__"

        val labelIndexer = new StringIndexer()
          .setInputCol(targetColumn)
          .setOutputCol(indexedLabel)
          .setHandleInvalid("keep")
          .fit(df)

        // Use this tranformer, because XGB_classifer doesn't work with DoubleType in target
        val doubleToIntTransformer = new DoubleToIntTransformer()
          .setInputCol(indexedLabel)
          .setOutputCol(indexedLabel)

        val featuresName = s"__${modelName}_features__"
        val featureAssembler = new VectorAssembler()
          .setInputCols(featureCols.toArray)
          .setOutputCol(featuresName)

        val afdf = featureAssembler.transform(df)
        val indexedFeatures = s"__indexed${featuresName}"
        val featureIndexer = new VectorIndexer()
          .setInputCol(featuresName)
          .setOutputCol(indexedFeatures)
          .setHandleInvalid("keep")
          .fit(afdf)

        val predictionName = s"__${modelName}_prediction__"

        val xgbParam = Map(
          "tracker_conf" -> TrackerConf(10000L, "scala")) // The timeout for all workers to connect to the tracker = 10s

        val xgb = new XGBoostClassifier(xgbParam)
          .setLabelCol(indexedLabel)
          .setFeaturesCol(indexedFeatures)
          .setPredictionCol(predictionName)
          .setRawPredictionCol("raw_prediction")
          .setProbabilityCol("probability_prediction")
          .setEta(learningRate)
          .setSeed(seed)
          .setTreeMethod(treeMethod)
          .setNumRound(numTrees)
          .setMaxDepth(maxDepth)
          .setLambda(lambda)
          .setAlpha(alpha)

        val labelConverter = new IndexToString()
          .setInputCol(predictionName)
          .setOutputCol(modelName + "_prediction")
          .setLabels(labelIndexer.labels)

        new Pipeline().setStages(
          Array(
            labelIndexer,
            doubleToIntTransformer,
            featureAssembler,
            featureIndexer,
            xgb,
            labelConverter))
      }

      val notPresentColumns = featureCols.diff(df.columns.toList)
      if (notPresentColumns.nonEmpty) {
        utils.sendError(
          searchId,
          s"Columns ${notPresentColumns.mkString(", ")} are not present in dataframe")
      }
      val p = createPipeline()
      val pModel = p.fit(df)
      var rdf = pModel.transform(df)
      val serviceCols =  rdf.columns.filter(_.matches("__.*__"))
      rdf = rdf.drop(serviceCols : _*)
      (pModel, rdf)
    }

}
