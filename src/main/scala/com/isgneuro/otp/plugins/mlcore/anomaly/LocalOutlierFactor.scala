package com.isgneuro.otp.plugins.mlcore.anomaly

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.outlier.LOF
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils

case class LocalOutlierFactor(featureCols: List[String], keywords: Map[String, String], id: Int, utils: PluginUtils) {

  import utils._

  val MIN_PTS_DEFAULT = 5

  /**
   *
   * @return input dataframe with added column for each input feature with computed LOF score
   *
   *         keywords:
   *  - min_pts: Minimum number of neighbours. Default: 5.
   *  - dist_type: The method to compute the distance between points. Default: euclidean. Out of use now.
   */

  def makePrediction(df: DataFrame): DataFrame = {

    val minPts = Caster.safeCast[Int](
      keywords.get("min_pts"),
      MIN_PTS_DEFAULT,
      sendError(id, "The value of parameter 'min_pts' should be of int type")
    )

    val featuresName = s"features"
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol(featuresName)

    val data = assembler.transform(df.drop("lof", "lof_result"))
    val indexedData = data.withColumn("lof_index", monotonically_increasing_id)

    val result = new LOF()
      .setMinPts(minPts)
      .setIndexCol("lof_index")
      .transform(indexedData)
      .withColumnRenamed("lof", "lof_result")

    val out = result.join(indexedData, usingColumns = Seq("lof_index"), joinType = "inner")
    out.drop("vector", "features", "lof_index")
  }
}

object LocalOutlierFactor extends ApplyModel {
  override def apply(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String], targetName: Option[String], keywords: Map[String, String], utils: PluginUtils): DataFrame => DataFrame =
    LocalOutlierFactor(featureCols, keywords, searchId, utils).makePrediction
}