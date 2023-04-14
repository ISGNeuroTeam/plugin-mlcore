package com.isgneuro.otp.plugins.mlcore.stat

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.DataFrame
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils
import org.apache.spark.ml.feature.QuantileDiscretizer

case class Bucket(featureCols: List[String], keywords: Map[String, String], searchId: Int, utils: PluginUtils) {
  import utils._

  val buckets: Int = Caster.safeCast[Int](
    keywords.get("bins"),
    2,
    sendError(searchId, "Cannot parse number of buckets")
  )

  def transform(df: DataFrame): DataFrame = {
    featureCols.foldLeft(df) {
      case(acc, col) =>
        val newcol = s"${col}_bin"
        val discretizer = new QuantileDiscretizer()
          .setInputCol(col)
          .setOutputCol(newcol)
          .setNumBuckets(buckets)
        discretizer.fit(acc).transform(acc)
    }
  }

}

object Bucket extends ApplyModel {
  override def apply(modelName: String,
                     modelConfig: Option[Config],
                     searchId: Int,
                     featureCols: List[String],
                     targetName: Option[String],
                     keywords: Map[String, String],
                     utils: PluginUtils): DataFrame => DataFrame =
    Bucket(featureCols, keywords, searchId, utils).transform
}
