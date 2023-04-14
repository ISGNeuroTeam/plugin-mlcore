package com.isgneuro.otp.plugins.mlcore.util

import scala.util.matching.Regex
import java.time.Instant

object Datetime {
  val getCurrentTimeInSeconds: () => Long = () => Instant.now.getEpochSecond
  val getCurrentTimeInMilliSeconds: () => Long = () => Instant.now.getEpochSecond * 1000
  val pattern_timespan = """(\d+)([a-z]+)"""
  val pattern_int = """(\d+)"""
  
  def getSpanInSeconds(spanText: String): Int = {
    val pattern_regex = new Regex(pattern_timespan)
    val pattern_regex(num, dim) = spanText
    val mult: Int = dim match {
      case "s" | "sec" => 1
      case "min" | "m" => 60
      case "h"         => 60 * 60
      case "d"         => 60 * 60 * 24
      case "w"         => 60 * 60 * 24 * 7
      case _           => 1
    }
    num.toInt * mult
  }

  def getSpanInMilliSeconds(spanText: String): Int = getSpanInSeconds(spanText) * 1000

  def getRelativeTime: (Long, String) => Long = (time: Long, relative: String) => {
    val mods = List("h", "m", "min", "s", "sec", "d", "w", "mon").mkString("|")
    val (mod, span, rnd) = s"""(\\-|\\+)((\\d+)($mods))(@($mods))?""".r
      .findAllIn(relative).matchData.map(x => (x.group(1), x.group(2), Option(x.group(6)))).toList.headOption.getOrElse((None, None, None))
    val shift = getSpanInSeconds(span.toString)
    val shiftDecimal = if (mod.toString == "+") shift else -shift
    rnd match {
      case Some(r) =>
        val rSpan = getSpanInSeconds(s"1$r")
        ((time + shiftDecimal) / rSpan).toInt * rSpan
      case _ => time + shiftDecimal
    }
  }
}
