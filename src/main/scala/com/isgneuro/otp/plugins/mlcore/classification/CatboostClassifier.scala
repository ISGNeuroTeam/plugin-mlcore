package com.isgneuro.otp.plugins.mlcore.classification

import com.typesafe.config.Config
import org.apache.spark.ml.feature.{ IndexToString, StringIndexer, VectorAssembler }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils
import ai.catboost.spark.CatBoostClassifier
import com.isgneuro.otp.plugins.mlcore.util.{Caster, CatboostFixer, ColRenamer}

import scala.util.{Failure, Success, Try}

object CatboostClassifier extends FitModel {

  val DEPTH_DEFAULT = 3
  val ITERATION_SUBSAMPLE_DEFAULT:Float = 1.0.toFloat
  val ITERATIONS_DEFAULT = 100
  val RANDOM_SEED_DEFAULT = 0
  val LEARNING_RATE_DEFAULT:Float = 0.03.toFloat
  val L2_LEAF_REG_DEFAULT:Float = 0.3.toFloat
  val BORDER_COUNT_DEFAULT = 254

  /** Catboost classifier algorithm.
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

      val iterationSubsample = Caster.safeCast[Float](
        keywords.get("iterationSubsample"),
        ITERATION_SUBSAMPLE_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'iterationSubsample' should be of float type")
      )

      val learningRate = Caster.safeCast[Float](
        keywords.get("learningRate"),
        LEARNING_RATE_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'learningRate' should be of float type")
      )

      val iterations = Caster.safeCast[Int](
        keywords.get("iterations"),
        ITERATIONS_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'iterations' should be of int type")
      )

      val depth = Caster.safeCast[Int](
        keywords.get("depth"),
        DEPTH_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'depth' should be of int type")
      )

      val l2LeafReg = Caster.safeCast[Float](
        keywords.get("l2LeafReg"),
        L2_LEAF_REG_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'l2LeafReg' should be of float type")
      )

      val borderCount = Caster.safeCast[Int](
        keywords.get("borderCount"),
        BORDER_COUNT_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'borderCount' should be of int type")
      )

      val seed: Long = Caster.safeCast(
        keywords.get("randomSeed"),
        RANDOM_SEED_DEFAULT,
        utils.sendError(searchId, "The value of parameter 'seed' should be of long type")
      )

      val labelIndexer = new StringIndexer()
        .setInputCol(targetColumn)
        .setOutputCol("label")
        .setHandleInvalid("skip")
        .fit(df)

      val featuresName = "features"
      val featureAssembler = new VectorAssembler()
        .setInputCols(featureCols.toArray)
        .setOutputCol(featuresName).setHandleInvalid("skip")

      val cc = new CatBoostClassifier()
        .setIterations(iterations)
        .setLearningRate(learningRate)
        .setDepth(depth)
        .setRsm(iterationSubsample)
        .setL2LeafReg(l2LeafReg)
        .setBorderCount(borderCount)
        .setRandomSeed(seed.toInt)

      val renamer = new ColRenamer("21", "probability", modelName+"_probabilities")

      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol(modelName + "_prediction")
        .setLabels(labelIndexer.labels)

      val fixer = new CatboostFixer("21", modelName+"_probabilities", modelName+"_probabilities")

      new Pipeline().setStages(Array(labelIndexer, featureAssembler, cc, renamer,
        labelConverter, fixer))
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