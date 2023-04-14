package com.isgneuro.otp.plugins.mlcore.feature

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import ot.dispatcher.plugins.small.sdk.FitModel
import ot.dispatcher.sdk.PluginUtils

case class PrincipalComponentAnalysis(fieldsUsed: List[String], modelName: String, keywords: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val N_COMPONENTS_DEFAULT = 2

  /**
   *
   * @return input dataframe with added pca columns
   *
   *         keywords:
   *  - n_components: The number of PCA components. Default: 2.
   */

  def makePrediction(inputDf: DataFrame): (PipelineModel, DataFrame) = {

    val nComponents = Caster.safeCast[Int](
      keywords.get("n_components"),
      N_COMPONENTS_DEFAULT,
      sendError(searchId, "The value of parameter 'n_components' should be of int type")
    )

    val dataFrame = inputDf.drop(inputDf.columns.filter(_.matches(".*pca.*")): _*)

    val featuresName = s"__${modelName}_features__"

    val featureAssembler = new VectorAssembler()
      .setInputCols(fieldsUsed.toArray)
      .setOutputCol(featuresName)

    val pcaFeatures = s"__pca_features__"
    val pca = new PCA()
      .setInputCol(featuresName)
      .setOutputCol(pcaFeatures)
      .setK(nComponents)

    val vectorToArrayUdf = udf { vec: Any =>
      vec match {
        case v: Vector => v.toArray
        case v => throw new IllegalArgumentException(
          "function vector_to_array requires a non-null input argument and input type must be Vector" +
            s"but got ${if (v == null) "null" else v.getClass.getName}.")
      }
    }.asNonNullable()

    val pipeline = new Pipeline().setStages(Array(featureAssembler, pca))
    val pipelineModel = pipeline.fit(dataFrame)
    val pcadf = pipelineModel.transform(dataFrame)
      .withColumn("temp", vectorToArrayUdf(col("__pca_features__"))).select(
      col("*") +: (0 until nComponents).map(i => col("temp").getItem(i).as(s"pca$i")): _*
    ).drop(s"__${modelName}_features__", "__pca_features__", "temp")
    (pipelineModel, pcadf)
  }
}

object PrincipalComponentAnalysis extends FitModel {

  override def fit(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String],
                   targetCol: Option[String], keywords: Map[String, String], utils: PluginUtils)
  : DataFrame => (PipelineModel, DataFrame) =

    df => {
      val result: (PipelineModel, DataFrame) = PrincipalComponentAnalysis(featureCols, modelName, keywords, searchId, utils).makePrediction(df)
      result
    }
}
