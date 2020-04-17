package com.strings.model.tree

import com.strings.model.ensemble.Loss

abstract class CalcInfoGain(val loss:Loss = null) {
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

class TaylorGain(override val loss:Loss) extends CalcInfoGain{
  def _gain(y:Array[Double], y_pred:Array[Double]):Double = {
    val grad = loss.gradient(y,y_pred)
    val nominator = math.pow(y.zip(grad).map(x=> x._1 * x._2).sum,2)
    val denominator = loss.hess(y, y_pred).sum
    0.5 * (nominator / denominator)
  }

  override def impurity_calculation(y: Array[Double], y1: Array[Double], y2: Array[Double]): Double = {
    val (yy, y_pred) = Utils.split(y)
   val (yy1, y1_pred) = Utils.split(y1)
   val (yy2, y2_pred) = Utils.split(y2)

    val true_gain = _gain(yy1, y1_pred)
    val false_gain = _gain(yy2, y2_pred)
    val gain = _gain(yy, y_pred)
    true_gain + false_gain - gain
  }
}
