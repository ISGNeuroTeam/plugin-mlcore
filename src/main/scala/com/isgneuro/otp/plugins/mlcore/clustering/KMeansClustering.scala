package com.isgneuro.otp.plugins.mlcore.clustering

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

import scala.util.{Failure, Success, Try}

case class KMeansClustering(fieldsUsed: List[String], modelName: String, properties: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val DISTANCE_DEFAULT = "euclidean"
  val MAX_ITER_DEFAULT = 300
  val SEED_DEFAULT = 0
  val TOL_DEFAULT = 0.0001
  val RANGE_DEFAULT = "2-20"
  val SUPPORTED_DISTANCES = List("euclidian", "cosine")

  /**
   *
   * @return input dataframe with added column cluster that shows defined cluster numbers
   *
   *         keywords:
   *  - num: The number of clusters to form as well as the number of centroids to generate. Default: 2.
   *  - max_iter: Maximum number of iterations of the k-means algorithm for a single run. Default: 300.
   *  - seed: The random seed. Use only for tests.
   *  - tol: Relative tolerance of the difference in the cluster centers of two consecutive iterations to declare
   *    convergence. Default: 0.0001.
   */

  def makePrediction(dataFrame: DataFrame): (PipelineModel, DataFrame) = {
    val kRange: Range.Inclusive = getRange("num")
    val distance: String = properties.get("distance") match {
      case Some(m) => if (SUPPORTED_DISTANCES.contains(m.stripPrefix("'").stripSuffix("'").trim)) m.stripPrefix("'").stripSuffix("'").trim
      else sendError(searchId, "The value of parameter 'distance' should be 'euclidean' or 'cosine'")
      case None => DISTANCE_DEFAULT
    }
    val maxIter = Caster.safeCast[Int](
      properties.get("max_iter"),
      MAX_ITER_DEFAULT,
      sendError(searchId, "The value of parameter 'max_iter' should be of int type")
    )
    val seed = Caster.safeCast[Long](
      properties.get("seed"),
      SEED_DEFAULT,
      sendError(searchId, "The value of parameter 'seed' should be of long type")
    )
    val tolCaster = Caster.safeCast[Double](
      properties.get("tol"),
      TOL_DEFAULT,
      sendError(searchId, "The value of parameter 'tol' should be of double type")
    )
    val tol = if (tolCaster < 0) sendError(searchId, "The value of parameter 'tol' should be non negative")
    else tolCaster

    val featuresName = s"__${modelName}_features__"
    val assembler = new VectorAssembler()
      .setInputCols(fieldsUsed.toArray)
      .setOutputCol(featuresName)
    val mDf = assembler.transform(dataFrame.drop("prediction"))

    val m: KMeansModel = null
    val SilhouetteMinValue: Double = -1.0
    val (_, bestModel, predictedDf) = kRange.foldLeft((SilhouetteMinValue, m, mDf)) { (best, k) =>
      val kmeans = new KMeans()
        .setK(k)
        .setFeaturesCol(featuresName)
        .setDistanceMeasure(distance)
        .setMaxIter(maxIter)
        .setTol(tol)
        .setSeed(seed)
      val model = kmeans.fit(mDf)
      val predictions = model.transform(mDf)
      val evaluator = new ClusteringEvaluator().setFeaturesCol(featuresName)
      val silhouette = evaluator.evaluate(predictions)
      if (silhouette >= best._1)
        (silhouette, model, predictions)
      else
        best
    }
    val pipeline = new Pipeline().setStages(Array(assembler, bestModel))
    val pipelineModel = pipeline.fit(dataFrame)

    (pipelineModel, predictedDf)
  }

  def getRange(name: String): Range.Inclusive = {
    val strInterval = properties.get(name) match {
      case Some(x) => x
      case None => RANGE_DEFAULT
    }
    Try(strInterval.toInt) match {
      case Success(d) => d to d
      case Failure(_) =>
        val arr = strInterval.split("-")
        Try(arr(0).toInt to arr(1).toInt) match {
          case Success(x) => x
          case Failure(_) => sendError(searchId, s"Value of $name param is not valid")
        }
    }
  }
}

object KMeansClustering extends FitModel {

  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      val result: (PipelineModel, DataFrame) = KMeansClustering(featureCols, modelName, keywords, searchId, utils).makePrediction(df)
      result
    }
}
