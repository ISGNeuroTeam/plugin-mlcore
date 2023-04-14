package com.isgneuro.otp.plugins.mlcore.regression

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

case class LinearRegression(featureCols: List[String], targetOption: Option[String], modelName: String, keywords: Map[String,String], searchId:Int, utils: PluginUtils) {
  import utils._

  val targetCol: String = targetOption.getOrElse(sendError(searchId, "Target column should be defined"))

  val REG_PARAM_DEFAULT = 0.0
  val ELASTICNET_PARAM_DEFAULT = 0.0
  val LOSS_DEFAULT = "huber"

  /**
   *
   * @return input dataframe with lr prediction
   *
   * keywords:
   *  - reg_param: Constant that multiplies the penalty terms. Default: 0.0.
   *  - reg: Regularization name. Can be 'lasso', 'ridge'. Elastic regularization os default when elasticnet_param is set.
   *  - elasticnet_param: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is
   *    an L2 penalty. Default: 0.0.
   */

  def createPipeline(): Pipeline = {
    val regCaster = Caster.safeCast[Double](
      keywords.get("reg_param"),
      REG_PARAM_DEFAULT,
      sendError(searchId, "The value of parameter 'reg_param' should be of double type")
    )
    val regParam = if (regCaster<0.0) sendError(searchId, "The value of parameter 'reg_param' should be non negative") else regCaster
    val reg = keywords.get("reg")
    val elasticnetCaster = Caster.safeCast[Double](
      keywords.get("elasticnet_param"),
      reg match {
        case Some("lasso") => 1.0
        case Some("ridge") => 0.0
        case None => ELASTICNET_PARAM_DEFAULT
        case _ => sendError(searchId, "The value of parameter 'reg' should be 'lasso' or 'ridge'")
      },
      sendError(searchId, "The value of parameter 'elasticnet_param' should be of double type")
    )
    val elasticnetParam = if (elasticnetCaster>1.0 || elasticnetCaster<0.0)
      sendError(searchId, "The value of parameter 'elasticnet_param' should be in [0.0; 1.0]") else elasticnetCaster

    val featuresName = s"__${modelName}_features__"
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol(featuresName)
    val lr = new org.apache.spark.ml.regression.LinearRegression()
      .setLabelCol(targetCol)
      .setFeaturesCol(featuresName)
      .setPredictionCol(modelName + "_prediction")
      .setRegParam(regParam)
      .setElasticNetParam(elasticnetParam)

    new Pipeline().setStages(Array(assembler, lr))
  }

  def prepareDf(dataFrame: DataFrame): Dataset[Row] = {
    dataFrame.drop(modelName + "_prediction").filter(targetCol +  " is not null")
  }

  def makePrediction(dataFrame: DataFrame): (PipelineModel, DataFrame) = {
    val p = createPipeline()
    val preparedDf = prepareDf(dataFrame)
    val pModel = p.fit(preparedDf)
    val rdf = pModel.transform(preparedDf).drop(s"__${modelName}_features__")
    (pModel, rdf)
  }
}

object LinearRegression extends FitModel {

  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      val result: (PipelineModel, DataFrame) = LinearRegression(featureCols, targetCol, modelName, keywords, searchId, utils).makePrediction(df)
      result
    }
}

