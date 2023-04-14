package com.isgneuro.otp.plugins.mlcore.commands

import org.apache.spark.sql.DataFrame
import ot.dispatcher.sdk.{PluginCommand, PluginUtils}
import ot.dispatcher.sdk.core.SimpleQuery

class OTLDescribe(query: SimpleQuery, utils: PluginUtils) extends PluginCommand(query, utils) {
  override def transform(_df: DataFrame): DataFrame = {
    _df.describe(returns.flatFields: _*)
  }
}

