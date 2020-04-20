package com.strings.model.ensemble

import com.strings.utils.Utils

trait Loss{
  def hess(pred:Array[Double],label:Array[Double]):Array[Double]
  def gradient(pred:Array[Double],label:Array[Double]):Array[Double]
  def transform(pred:Array[Double]):Array[Double]
}

object SquareLoss extends Loss{
  override def hess(pred: Array[Double], label: Array[Double]): Array[Double] = {
    Array.fill(label.length)(1.0)
  }

  override def gradient(pred: Array[Double], label: Array[Double]): Array[Double] = {
    label.zip(pred).map(x => -(x._1 - x._2))
  }

  override def transform(pred: Array[Double]): Array[Double] = pred
}

object LogisticLoss extends Loss{
  override def hess(pred: Array[Double], label: Array[Double]): Array[Double] = {
    val  pred1 = transform(pred)
    val  ret = new Array[Double](pred.length)
    for(i<- 0 until ret.length){
      ret(i) = pred1(i) * (1.0 - pred1(i))
    }
    ret
  }

  override def gradient(pred: Array[Double], label: Array[Double]): Array[Double] = {
    val pred1 = transform(pred)
    val ret = new Array[Double](pred1.length)
    for(i <- 0 until pred1.length) {
      ret(i) = pred1(i) - label(i)
    }
    ret
  }

  override def transform(pred: Array[Double]): Array[Double] = {
    val ret = new Array[Double](pred.length)
    for(i <- 0 until pred.length) {
      ret(i) = Utils.clip(1.0 / (1.0 + math.exp(-pred(i))))
    }
    ret
  }
}
