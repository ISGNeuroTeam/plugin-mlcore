package com.isgneuro.otp.plugins.mlcore.commands

import com.isgneuro.otp.plugins.mlcore.util.Caster
import org.apache.spark.sql.DataFrame
import ot.dispatcher.sdk.{PluginCommand, PluginUtils}
import ot.dispatcher.sdk.core.SimpleQuery
import ot.dispatcher.sdk.core.extensions.StringExt._

class OTLSample(query: SimpleQuery, utils: PluginUtils) extends PluginCommand(query, utils) {

  private val DEFAULT_SEED: Int = 42
  private val DEFAULT_FRACTION: Double = 0.1

  private val seed: Int = Caster.safeCast[Int](
    getKeyword("seed"),
    DEFAULT_SEED,
    sendError("Cannot parse seed to int")
  )

  private val fraction: Double = Caster.safeCast[Double](
    returns.flatFields.headOption.map(_.stripBackticks),
    DEFAULT_FRACTION,
    sendError("Cannot parse fraction")
  )

  override val fieldsUsed: List[String] = List.empty[String]

  override def transform(_df: DataFrame): DataFrame = {
    _df.sample(fraction, seed)
  }
}
