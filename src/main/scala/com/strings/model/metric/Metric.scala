package com.strings.model.metric

object Metric {

  def accuracy(pred:Array[Double],label:Array[Double]):Double = {
    val acc = pred.zip(label).map(x => if(x._1 == x._2) 1 else 0 )
    acc.sum.toDouble / pred.length
  }

  def accuracyReg(pred:Array[Double],label:Array[Double]):Double = {
      val relative = label.zip(pred).map(x => (x._1 - x._2)/(x._2 + 1e-16))
      val power = relative.map(x => x * x).sum / relative.size
    1 - math.sqrt(power)
  }

  def mape(pred:Array[Double],label:Array[Double]):Double = {
    val n = pred.length
    label.zip(pred).map(x => (x._1 - x._2)/(x._2 + 1e-16)).map(math.abs).sum / n
  }

  def rmse(pred:Array[Double],label:Array[Double]):Double = {
    val n = pred.length
    val relative = label.zip(pred).map(x => x._1 - x._2)
    math.sqrt(relative.map(math.pow(_,2)).sum / n)
  }
}
