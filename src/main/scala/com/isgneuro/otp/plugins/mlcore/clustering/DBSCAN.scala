package com.isgneuro.otp.plugins.mlcore.clustering

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.alitouka.spark.dbscan.spatial.Point
import org.alitouka.spark.dbscan.{Dbscan, DbscanSettings}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils

case class DBSCAN(fieldsUsed: List[String], keywords: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val EPS_DEFAULT = 10.0
  val MIN_PTS_DEFAULT = 5

  /**
   *
   * @param inputDf should has two num column
   * @return input dataframe with added column cluster that shows defined cluster numbers, 0 for noise points
   * @see https://github.com/alitouka/spark_dbscan/
   *
   *      keywords:
   *  - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default: 10.0
   *  - minPts: The number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself. Default: 5
   */

  def makePrediction(inputDf: DataFrame): DataFrame = {

    val eps = Caster.safeCast[Double](
      keywords.get("eps"),
      EPS_DEFAULT,
      sendError(searchId, "The value of parameter 'eps' should be of double type")
    )
    val minPts = Caster.safeCast[Int](
      keywords.get("min_pts"),
      MIN_PTS_DEFAULT,
      sendError(searchId, "The value of parameter 'min_pts' should be of int type")
    )

    val dataFrame = inputDf.drop("cluster")

    val rowDF = dataFrame.select(fieldsUsed.head, fieldsUsed.tail: _*).rdd
    val mat = rowDF.map(_.toSeq.toArray)
    val data: RDD[Point] = mat.map(line => new Point(line.map(_.toString.toDouble)))

    val clusteringSettings = new DbscanSettings().withEpsilon(eps).withNumberOfPoints(minPts)
    val model = Dbscan.train(data, clusteringSettings)
    val result = model.allPoints.map(x => Row(x.coordinates(0), x.coordinates(1), x.clusterId, x.pointId))
    val sc = dataFrame.sparkSession
    val schema = StructType(
      StructField("x", DoubleType, nullable = false) ::
        StructField("y", DoubleType, nullable = false) ::
        StructField("cluster", LongType, nullable = false) ::
        StructField("id", LongType) :: Nil)

    val df = sc.createDataFrame(result, schema)

    val dataFrame2 = dataFrame.withColumn("id", (monotonically_increasing_id + 1) * 10)

    val output = df.join(dataFrame2, usingColumn = "id").drop("id", "x", "y")
    output
  }
}

object DBSCAN extends ApplyModel {
  override def apply(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String], targetName: Option[String], keywords: Map[String, String], utils: PluginUtils): DataFrame => DataFrame =
    DBSCAN(featureCols, keywords, searchId, utils).makePrediction
}
