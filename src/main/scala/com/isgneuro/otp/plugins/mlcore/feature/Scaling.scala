package com.isgneuro.otp.plugins.mlcore.feature

import com.isgneuro.otp.plugins.mlcore.util.{ Caster, VectorDisassembler }
import com.typesafe.config.Config
import org.apache.spark.ml.feature.{MinMaxScaler, StandardScaler, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

case class Scaling(featureCols: List[String], keywords: Map[String, String], modelName: String, id: Int, utils: PluginUtils) {

  import utils._

  val WITH_MEAN_DEFAULT = true
  val WITH_STD_DEFAULT = true
  val MIN_DEFAULT = 0.0
  val MAX_DEFAULT = 1.0
  val SUPPORTED_SCALERS = List("standard", "minmax")

  /**
   *
   * @return input dataframe with scaled features
   *
   *         keywords:
   *  - with_mean: if true, the standard scaling is done with centering. Default: true.
   *  - with_std: if true, the standard scaling is done with scaling. Default: true.
   *  - min: minimum for minmaxscaler. Default: 0.0
   *  - max: maximum for minmaxscaler. Default: 1.0
   *  - scaler: the scaler name to use, standard or minmax. Default: standard.
   */

  def createPipeline(): Pipeline = {
    val withMean = Caster.safeCast[Boolean](
      keywords.get("with_mean"),
      WITH_MEAN_DEFAULT,
      sendError(id, "The value of parameter 'with_mean' should be of boolean type")
    )
    val withStd = Caster.safeCast[Boolean](
      keywords.get("with_std"),
      WITH_STD_DEFAULT,
      sendError(id, "The value of parameter 'with_std' should be of boolean type")
    )
    val min = Caster.safeCast[Double](
      keywords.get("min"),
      MIN_DEFAULT,
      sendError(id, "The value of parameter 'min' should be of double type")
    )
    val max = Caster.safeCast[Double](
      keywords.get("max"),
      MAX_DEFAULT,
      sendError(id, "The value of parameter 'max' should be of double type")
    )

    val featuresName = s"__${modelName}_features__"
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol(featuresName)

    val standard_scaler = new StandardScaler()
      .setInputCol(featuresName)
      .setOutputCol("scaledFeatures")
      .setWithStd(withStd)
      .setWithMean(withMean)

    val min_max_scaler = new MinMaxScaler()
      .setInputCol(s"__${modelName}_features__")
      .setOutputCol("scaledFeatures")
      .setMax(max)
      .setMin(min)

    val disassembler = new VectorDisassembler()
      .setInputCol("scaledFeatures")
      .setOutputCols(featureCols.toArray.map(_ + "_scaled"))

    val scaler = keywords.get("scaler") match {
      case Some(x) => x match {
        case "standard" => standard_scaler
        case "minmax" => min_max_scaler
        case _ => standard_scaler
      }
      case None => standard_scaler
    }

    new Pipeline().setStages(Array(assembler, scaler, disassembler))
  }

  def makePrediction(dataFrame: DataFrame): (PipelineModel, DataFrame) = {
    val p = createPipeline()
    val pModel = p.fit(dataFrame)
    val out = pModel.transform(dataFrame).drop(s"__${modelName}_features__")
    (pModel, out)
  }
}

object Scaling extends FitModel {

  override def fit(modelName: String,
                   modelConfig: Option[Config],
                   searchId: Int,
                   featureCols: List[String],
                   targetCol: Option[String],
                   keywords: Map[String, String],
                   utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      Scaling(featureCols, keywords, modelName, searchId, utils).makePrediction(df)
    }
}

