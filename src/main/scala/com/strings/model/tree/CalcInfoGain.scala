package com.strings.model.tree

abstract class CalcInfoGain {
  def impurity_calculation(y:Array[Double],y1:Array[Double],y2:Array[Double]):Double
}

object EntropyCalcGain extends CalcInfoGain{
  override def impurity_calculation(y: Array[Double], y1: Array[Double], y2: Array[Double]): Double = {
    val p:Double = y1.size.toDouble / y.size
    val entropy = Utils.calculate_entropy(y)
    val info_gain = entropy - p * Utils.calculate_entropy(y1) - (1- p ) * Utils.calculate_entropy(y2)
    info_gain
  }
}

object VarianceCalcGain extends CalcInfoGain{
  override def impurity_calculation(y: Array[Double], y1: Array[Double], y2: Array[Double]): Double = {
    val variance = Utils.calculate_variance(y)
    val p = y1.size.toDouble / y.size
    variance - p * Utils.calculate_variance(y1) - (1 - p) * Utils.calculate_variance(y2)
  }
}
