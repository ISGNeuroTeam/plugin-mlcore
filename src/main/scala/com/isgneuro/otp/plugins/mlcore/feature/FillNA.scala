package com.isgneuro.otp.plugins.mlcore.feature

import com.isgneuro.otp.plugins.mlcore.util.Caster
import com.typesafe.config.Config
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, first, last, when}
import org.apache.spark.sql.types.{NumericType, StringType}
import org.apache.spark.sql.{Column, DataFrame}
import ot.dispatcher.plugins.small.sdk.ApplyModel
import ot.dispatcher.sdk.PluginUtils

import scala.util.{Failure, Success, Try}

case class FillNA(fieldsUsed: List[String], keywords: Map[String, String], searchId: Int, utils: PluginUtils) {

  import utils._

  val MISSING_VALUE_DEFAULT: Double = Double.NaN
  val NEW_COLUMNS_DEFAULT: Boolean = false
  val STRATEGY_DEFAULT = "ffill"
  val FILL_VALUE_DEFAULT = 0.0
  val FILL_VALUE_STRING_DEFAULT = "missing"
  val SUPPORTED_STRATEGIES = List("drop", "ffill", "pad", "filldown", "bfill", "backfill", "const")
  val DROP_STRATEGY_DEFAULT = "any"
  val SUPPORTED_DROP_STRATEGIES = List("any", "all")

  /**
   *
   * @return input dataframe with added pca columns
   *
   *         keywords:
   *  - missing_value: The placeholder for the missing values. All occurrences of missingValue will be imputed.
   *    Note that null values are always treated as missing. Default: Double.NaN
   *  - fill_value: When strategy == “const”, fill_value is used to replace all occurrences of missing_values.
   *    If left to the default, fill_value will be 0.0 when imputing numerical data, "missing" when imputing string data.
   *  - strategy: The imputation strategy. Defaul: ffill (fill down)
   *  - drop_strategy: The drop strategy. Whe  strategy == "drop", if all, drop only rows with all values missing,
   *    if any, drop rows with any values missing. Default: any.
   *  - new_columns: If true, output columns will be added with names of input columns + "_imputed";
   *    if false, filling is done on input columns. Default: false.
   */

  def makePrediction(inputDf: DataFrame): DataFrame = {

    val missingValue = keywords.get("missing_value") match {
      case Some(m) => Try(m.toDouble) match {
        case Success(x) => x
        case Failure(_) => m
      }
      case None => MISSING_VALUE_DEFAULT
    }
    val strategy = keywords.get("strategy") match {
      case Some(m) if SUPPORTED_STRATEGIES.contains(m) => m
      case Some(_) => utils.sendError(searchId, s"""No such strategy. Available strategies: ${SUPPORTED_STRATEGIES.mkString(", ")}.""")
      case None => STRATEGY_DEFAULT
    }
    val dropStrategy = keywords.get("drop_strategy") match {
      case Some(m) if SUPPORTED_DROP_STRATEGIES.contains(m) => m
      case Some(_) => utils.sendError(searchId, "No such strategy. Available strategies: any, all.")
      case None => DROP_STRATEGY_DEFAULT
    }
    val fillValue = keywords.get("fill_value") match {
      case Some(m) => Try(m.toDouble) match {
        case Success(x) => x
        case Failure(_) => m
      }
      case None => if (missingValue.isInstanceOf[Double]) FILL_VALUE_DEFAULT else FILL_VALUE_STRING_DEFAULT
    }
    val newColumns = Caster.safeCast[Boolean](
      keywords.get("new_columns"),
      NEW_COLUMNS_DEFAULT,
      sendError(searchId, "The value of parameter 'new_columns' should be of boolean type")
    )

    // Use only in ffill/bfill methods
    // Requirement: names of 'by' columns must contain only letters, digits, and underscores
    val byCols = keywords.get("by").map(_.split(","))

    val outputColumns = fieldsUsed.toArray.map(_ + "_imputed")
    val df = inputDf.drop(outputColumns: _*)

    Try(fillValue.toString.toDouble) match {
      case Success(_) => fieldsUsed.foreach(column => require(df.schema(column).dataType.isInstanceOf[NumericType],
        s"Input columns must be of numeric type but got ${df.schema(column).dataType} for column $column"))
      case Failure(_) => fieldsUsed.foreach(column => require(df.schema(column).dataType.isInstanceOf[StringType],
        s"Input columns must be of string type but got ${df.schema(column).dataType} for column $column"))

    }

    // newCols is an array of copies of input columns with missingValue changed to null.
    // newCols are using for window operation for ffill and bfill strategies and adding new columns for const strategy.
    val newCols: Array[Column] = fieldsUsed.toArray.map {
      inputCol =>
        val inputType = df.schema(inputCol).dataType
        when(col(inputCol) === missingValue, null)
          .otherwise(col(inputCol))
          .cast(inputType)
    }
    val partitionsNum = df.rdd.getNumPartitions
    strategy match {
      case "ffill" | "pad" | "filldown" | "bfill" | "backfill" =>
        val imputedDf = strategy match {
          case "ffill" | "pad" | "filldown" =>
            val window = byCols match {
              case Some(cols) => Window.partitionBy(cols.map(col): _*).rowsBetween(Window.unboundedPreceding, 0)
              case _ => Window.rowsBetween(Window.unboundedPreceding, 0)
            }
            fieldsUsed.zipWithIndex.foldLeft(df.toDF()) {
              (acc, field) => acc.withColumn(field._1 + "_imputed", last(newCols(field._2), ignoreNulls = true).over(window))
            }
          case "bfill" | "backfill" =>
            val window = byCols match {
              case Some(cols) => Window.partitionBy(cols.map(col): _*).rowsBetween(0, Window.unboundedFollowing)
              case _ => Window.rowsBetween(0, Window.unboundedFollowing)
            }
            fieldsUsed.zipWithIndex.foldLeft(df.toDF()) {
              (acc, field) => acc.withColumn(field._1 + "_imputed", first(newCols(field._2), ignoreNulls = true).over(window))
            }
        }
        val partitionedDf = imputedDf.repartition(partitionsNum)
        if (!newColumns) {
          val outDf = partitionedDf.drop(fieldsUsed: _*)
          val renamedCols = fieldsUsed.zip(fieldsUsed.map(_ + "_imputed"))
          renamedCols.foldLeft(outDf)((acc, c) => acc.withColumnRenamed(c._2, c._1))
        }
        else partitionedDf
      case "drop" =>
        df.na.replace(fieldsUsed, Map((missingValue, null)))
          .na.drop(dropStrategy, fieldsUsed)
      case "const" =>
        val (outDf, nullColumns) =
          if (!newColumns) (df.na.replace(fieldsUsed, Map((missingValue, null))), fieldsUsed)
          else (fieldsUsed.zipWithIndex.foldLeft(df.toDF()) { (acc, field) =>
            acc.withColumn(field._1 + "_imputed", newCols(field._2))
          }, fieldsUsed.map(_ + "_imputed"))
        if (df.toDF().schema(fieldsUsed.head).dataType.isInstanceOf[StringType]) {
          val fill = fillValue.toString
          outDf.na.fill(fill, nullColumns)
        }
        else {
          val fill = fillValue.toString.toDouble
          outDf.na.fill(fill, nullColumns)
        }
    }
  }
}

object FillNA extends ApplyModel {

  override def apply(modelName: String, modelConfig: Option[Config], searchId: Int, featureCols: List[String], targetName: Option[String], keywords: Map[String, String], utils: PluginUtils): DataFrame => DataFrame =
    FillNA(featureCols, keywords, searchId, utils).makePrediction
}
