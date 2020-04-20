package com.strings.model.tree

import com.strings.model.ensemble.Loss
import com.strings.utils.Utils

abstract class CalcInfoGain(val loss:Loss = null) {
  def impurity_calculation(y:Array[Double],y1:Array[Double],y2:Array[Double]):Double
}

object EntropyCalcGain extends CalcInfoGain{
  override def impurity_calculation(y: Array[Double], y1: Array[Double], y2: Array[Double]): Double = {
    val p:Double = y1.length.toDouble / y.length
    val entropy = Utils.calculate_entropy(y)
    val info_gain = entropy - p * Utils.calculate_entropy(y1) - (1- p ) * Utils.calculate_entropy(y2)
    info_gain
  }
}

object VarianceCalcGain extends CalcInfoGain{
  override def impurity_calculation(y: Array[Double], y1: Array[Double], y2: Array[Double]): Double = {
    val variance = Utils.calculate_variance(y)
    val p = y1.length.toDouble / y.length
    variance - p * Utils.calculate_variance(y1) - (1 - p) * Utils.calculate_variance(y2)
  }
}

