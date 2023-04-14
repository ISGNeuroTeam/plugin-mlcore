package com.isgneuro.otp.plugins.mlcore.classification

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

import scala.util.{Failure, Success, Try}

case class LogisticRegression(featureCols: List[String], targetOption: Option[String], modelName: String, keywords: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val targetCol: String = targetOption.getOrElse(sendError(searchId, "Target column should be defined"))

  val REG_PARAM_DEFAULT = 0.0
  val ELASTICNET_PARAM_DEFAULT = 0.0
  val FAMILY_DEFAULT = "auto"
  val INTERCEPT_DEFAULT = true
  val STANDARDIZATION_DEFAULT = true
  val MAX_ITER_DEFAULT = 300
  val THRESHOLD_DEFAULT = 0.5
  val THRESHOLDS_DEFAULT: Array[Double] = Array[Double]()
  val SUPPORTED_FAMILIES = List("auto", "binomial", "multinomial")

  /**
   *
   * @return input dataframe with log_reg class prediction
   *
   *         keywords:
   *  - reg_param: Constant that multiplies the penalty terms. Default: 0.0.
   *  - reg: Regularization name. Can be 'lasso', 'ridge'. Elastic regularization os default when elasticnet_param is set.
   *  - elasticnet_param: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is
   *    an L2 penalty. Default: 0.0.
   *  - family: Param for the name of family which is a description of the label distribution to be used in the model. Default: auto.
   *  - fit_intercept: Param for whether to fit an intercept term. Default: true.
   *  - standardization: Param for whether to standardize the training features before fitting the model. Default: true.
   *  - max_iter: Param for maximum number of iterations (>= 0). Default: 300.
   *  - threshold: Param for threshold in binary classification prediction, in range [0, 1]. Default: 0.5.
   *  - thresholds: Param for Thresholds in multi-class classification to adjust the probability of predicting each class.
   */

  def createPipeline(dataFrame: DataFrame): Pipeline = {
    val regCaster = Caster.safeCast[Double](
      keywords.get("reg_param"),
      REG_PARAM_DEFAULT,
      sendError(searchId, "The value of parameter 'reg_param' should be of double type")
    )
    val regParam = if (regCaster < 0.0) sendError(searchId, "The value of parameter 'reg_param' should be non negative") else regCaster
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
    val elasticnetParam = if (elasticnetCaster > 1.0 || elasticnetCaster < 0.0)
      sendError(searchId, "The value of parameter 'elasticnet_param' should be in [0.0; 1.0]") else elasticnetCaster
    val family = keywords.get("family") match {
      case Some(m) => if (SUPPORTED_FAMILIES.contains(m.stripPrefix("'").stripSuffix("'").trim)) m.stripPrefix("'").stripSuffix("'").trim
      else sendError(searchId, "The value of parameter 'family' should be in 'auto', 'binomial', 'multinomial'")
      case None => FAMILY_DEFAULT
    }
    val fitIntercept = Caster.safeCast[Boolean](
      keywords.get("fit_intercept"),
      INTERCEPT_DEFAULT,
      sendError(searchId, "The value of parameter 'fit_intercept' should be of boolean type")
    )
    val standardization = Caster.safeCast[Boolean](
      keywords.get("standardization"),
      STANDARDIZATION_DEFAULT,
      sendError(searchId, "The value of parameter 'standardization' should be of boolean type")
    )
    val maxIter = Caster.safeCast[Int](
      keywords.get("max_iter"),
      MAX_ITER_DEFAULT,
      sendError(searchId, "The value of parameter 'max_iter' should be of int type")
    )
    val thresholdCaster = Caster.safeCast[Double](
      keywords.get("threshold"),
      THRESHOLD_DEFAULT,
      sendError(searchId, "The value of parameter 'threshold' should be of double type")
    )
    val threshold = if (thresholdCaster < 0.0 | thresholdCaster > 1.0)
      sendError(searchId, "The value of parameter 'threshold' should be in range [0.0;1.0]") else thresholdCaster
    val thresholds = keywords.get("thresholds") match {
      case Some(m) => Try(m.split(",").map(x => Try(x.toDouble).getOrElse(0.5)).distinct) match {
        case Success(n) => if (n.exists(x => (x < 0.0) | (x > 1.0))) sendError(searchId, "The values of parameter 'thresholds' should be in range [0.0;1.0]") else n
        case Failure(_) => sendError(searchId, "The value of parameter 'thresholds' should be of double array type")
      }
      case None => THRESHOLDS_DEFAULT
    }

    val featuresName = s"__${modelName}_features__"
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol(featuresName)
    val indexedLabel = s"__indexed${targetCol}__"

    val labelIndexer = new StringIndexer()
      .setInputCol(targetCol)
      .setOutputCol(indexedLabel)
      .setStringOrderType("alphabetAsc")
      .fit(dataFrame)

    val lr1 = new org.apache.spark.ml.classification.LogisticRegression()
      .setLabelCol(indexedLabel)
      .setFeaturesCol(featuresName)
      .setPredictionCol(modelName + "_index_prediction")
      .setRegParam(regParam)
      .setElasticNetParam(elasticnetParam)
      .setFamily(family)
      .setMaxIter(maxIter)
      .setFitIntercept(fitIntercept)
      .setStandardization(standardization)
      .setThreshold(threshold)
    val lr = if (thresholds.length > 0) lr1.setThresholds(thresholds) else lr1
    val labelConverter = new IndexToString()
      .setInputCol(modelName + "_index_prediction")
      .setOutputCol(modelName + "_prediction")
      .setLabels(labelIndexer.labels)

    new Pipeline().setStages(Array(assembler, labelIndexer, lr, labelConverter))
  }

  def prepareDf(dataFrame: DataFrame): Dataset[Row] = {
    dataFrame.drop(modelName + "_prediction").filter(targetCol + " is not null")
  }

  def makePrediction(dataFrame: DataFrame): (PipelineModel, DataFrame) = {
    val vecToArray = udf((xs: linalg.Vector) => xs.toArray)
    val p = createPipeline(dataFrame)
    val df = prepareDf(dataFrame)
    val pModel = p.fit(df)
    val rdf = pModel.transform(df).drop(s"__${modelName}_features__")
    val out = rdf.withColumn("probabilities", vecToArray(rdf("probability")))
    (pModel, out.drop(out.columns.filter(_.matches(".*index.*")): _*).drop("rawPrediction", "probability"))
  }
}

object LogisticRegression extends FitModel {

  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      val result: (PipelineModel, DataFrame) = LogisticRegression(featureCols, targetCol, modelName, keywords, searchId, utils).makePrediction(df)
      result
    }
}



