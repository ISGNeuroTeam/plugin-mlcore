package com.isgneuro.otp.plugins.mlcore.commands

import com.isgneuro.otp.plugins.mlcore.util.Caster
import org.apache.spark
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.{DataFrame, functions}
import com.isgneuro.otp.plugins.mlcore.ts.TimeSeriesPredictor.FUTURE_DEFAULT
import ot.dispatcher.sdk.{PluginCommand, PluginUtils}
import ot.dispatcher.sdk.core.SimpleQuery
import ot.dispatcher.sdk.core.extensions.DataFrameExt._
import ot.dispatcher.sdk.core.functions.Datetime

class OTLMakefuture(query: SimpleQuery, utils: PluginUtils) extends PluginCommand(query, utils){
  private val FUTURE_DEFAULT = 1
  val future: Int = Caster.safeCast[Int](
    getKeyword("count"),
    FUTURE_DEFAULT,
    sendError("The value of parameter 'future' should be of int type")
  )

  val span: Int = Datetime.getSpanInSeconds(getKeyword("span").getOrElse("86400"))

  override def transform(_df: DataFrame): DataFrame = {
    val maxTime = _df.agg(functions.max("_time").cast(LongType)).collect()(0).getLong(0)
    val df_append = _df.sparkSession.range(1, future + 1)
      .withColumn("_time", lit(span) * col("id") + lit(maxTime))
      .drop("id")
    _df.append(df_append)
  }
}