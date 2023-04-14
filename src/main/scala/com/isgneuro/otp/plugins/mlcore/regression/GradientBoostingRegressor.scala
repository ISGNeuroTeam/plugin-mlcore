package com.isgneuro.otp.plugins.mlcore.regression

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

import scala.util.{Failure, Success, Try}

object GradientBoostingRegressor extends FitModel {

  val MAX_DEPTH_DEFAULT = 3
  val ITERATION_SUBSAMPLE_DEFAULT = 1
  val MIN_INFO_GAIN_DEFAULT = 0.0
  val MIN_LEAF_SAMPLES_DEFAULT = 1
  val NUM_TREES_DEFAULT = 100
  val MAX_BINS_DEFAULT = 32
  val SUBSET_STRATEGY_DEFAULT = "auto"
  val LOSS_TYPE_DEFAULT = "squared"
  val LEARNING_RATE_DEFAULT = 0.1
  val RANDOM_SEED_DEFAULT = 0
  val SUPPORTED_STRATEGIES = List("auto", "all", "onethird", "sqrt", "log2")
  val SUPPORTED_LOSSES = List("absolute", "squared")

  /** Gradient boosting regressor algorithm.
   */

  override def fit(modelName: String,
                   modelConfig: Option[Config],
                   searchId: Int,
                   featureCols: List[String],
                   targetCol: Option[String],
                   keywords: Map[String, String],
                   utils: PluginUtils):
  DataFrame => (PipelineModel, DataFrame) =
    df => {

      def createPipeline() = {

        val targetColumn = Try(targetCol.get) match{
          case Success(n) => n
          case Failure(_) => utils.sendError(searchId, "The name of a target column was not set")
        }

        val maxDepth = Caster.safeCast[Int](
          keywords.get("maxDepth"),
          MAX_DEPTH_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'maxDepth' should be of int type")
        )

        val learningRate = Caster.safeCast[Double](
          keywords.get("learningRate"),
          LEARNING_RATE_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'learningRate' should be of float type")
        )

        val iterationSubsample = Caster.safeCast[Float](
          keywords.get("iterationSubsample"),
          ITERATION_SUBSAMPLE_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'iterationSubsample' should be of float type")
        )

        val minInfoGain = Caster.safeCast[Double](
          keywords.get("minInfoGain"),
          MIN_INFO_GAIN_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'minInfoGain' should be of float type")
        )

        val minLeafSamples = Caster.safeCast[Int](
          keywords.get("minLeafSamples"),
          MIN_LEAF_SAMPLES_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'minLeafSamples' should be of int type")
        )

        val numTrees = Caster.safeCast[Int](
          keywords.get("numTrees"),
          NUM_TREES_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'numTrees' should be of int type")
        )

        val maxBins = Caster.safeCast[Int](
          keywords.get("maxBins"),
          MAX_BINS_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'maxBins' should be of int type")
        )

        val subsetStrategy = keywords.get("subsetStrategy") match{
          case Some(m) if SUPPORTED_STRATEGIES.contains(m) => m
          case Some(_) => utils.sendError(searchId, "No such strategy. Available strategies: auto, all, onethird, sqrt, log2")
          case None => SUBSET_STRATEGY_DEFAULT
        }

        val lossType = keywords.get("lossType") match{
          case Some(m) if SUPPORTED_LOSSES.contains(m) => m
          case Some(_) => utils.sendError(searchId, "No such impurity. Available impurities: gini, entropy")
          case None => LOSS_TYPE_DEFAULT
        }

        val seed: Long = Caster.safeCast(
          keywords.get("randomSeed"),
          RANDOM_SEED_DEFAULT,
          utils.sendError(searchId, "The value of parameter 'seed' should be of long type")
        )

        val featuresName = s"__${modelName}_features__"
        val featureAssembler = new VectorAssembler()
          .setInputCols(featureCols.toArray)
          .setOutputCol(featuresName)

        val predictionName = s"${modelName}_prediction"
        val gb = new GBTRegressor()
          .setLabelCol(targetColumn)
          .setFeaturesCol(featuresName)
          .setPredictionCol(predictionName)
          .setMinInfoGain(minInfoGain)
          .setMinInstancesPerNode(minLeafSamples)
          .setStepSize(learningRate)
          .setMaxDepth(maxDepth)
          .setSubsamplingRate(iterationSubsample)
          .setMaxIter(numTrees)
          .setMaxBins(maxBins)
          .setFeatureSubsetStrategy(subsetStrategy)
          .setLossType(lossType)
          .setSeed(seed)

        new Pipeline().setStages(Array(featureAssembler, gb))
      }

      val notPresentColumns = featureCols diff df.columns.toList
      if (notPresentColumns.nonEmpty) {
        utils.sendError(searchId, s"Columns ${notPresentColumns.mkString(", ")} are not present in dataframe")
      }
      val p = createPipeline()
      val df1 = df.drop(modelName + "_prediction")
      val pModel = p.fit(df1)
      var rdf = pModel.transform(df1)
      val serviceCols =  rdf.columns.filter(_.matches("__.*__"))
      rdf = rdf.drop(serviceCols : _*)
      (pModel, rdf)
    }
}
