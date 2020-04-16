package com.strings.loss

import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.sigmoid

abstract class LossFunc(val alpha:Double = 0.5,
                        val l1_ratio:Double=0.5) extends Serializable{
  def sign(d: Double): Double ={
    if(d > 0.0) 1.0
    else if(d < 0.0) -1.0
    else 0.0
  }
  def costNgradient(training: BDM[Double],label:BDV[Double],weight:BDV[Double]):(Double,BDV[Double])

}

class LogisticLoss extends LossFunc{
  override def costNgradient(training: BDM[Double], label: BDV[Double], weights: BDV[Double]): (Double, BDV[Double]) ={
    val raw_output = (training * weights).map(sigmoid(_))
    val error = raw_output - label
    val loss: Double = error.t * error
    val grad = (training.t * error) :/ training.rows.toDouble
    (loss, grad)
  }
}

class SquareLoss extends LossFunc{
  override def costNgradient(training: BDM[Double], label: BDV[Double], weights: BDV[Double]): (Double, BDV[Double]) ={
    val raw_output = training * weights
    val error = raw_output - label
    val m = label.length
    val loss: Double = error.t * error / (2 * m)
    val grad = (error.t * training).t
    (loss, grad.map(x => x/m))
  }
}

class SquareLossWithL1(override val alpha:Double = 0.5,
                       override val l1_ratio:Double=0.5) extends LossFunc {



  override def costNgradient(training: BDM[Double], label: BDV[Double], weights: BDV[Double]): (Double, BDV[Double]) = {
    val raw_output = training * weights
    val error = raw_output - label
    val m = label.length
    val l1_loss = alpha * sum(weights.map(math.abs(_)))
    val loss: Double = error.t * error / (2 * m) + l1_loss
    val l1_grad = weights.map(sign(_)) :* alpha
    val grad = (error.t * training).t + l1_grad
    (loss, grad.map(x => x / m))
  }
}

class SquareLossWithL2(override val alpha:Double = 0.5,
                       override val l1_ratio:Double=0.5) extends LossFunc {

  override def costNgradient(training: BDM[Double], label: BDV[Double], weights: BDV[Double]): (Double, BDV[Double]) = {
    val raw_output = training * weights
    val error = raw_output - label
    val m = label.length
    val l2_loss = alpha * sum(weights.map(x => x * x))
    val loss: Double = error.t * error / (2 * m) + l2_loss
    val l2_grad = weights :* alpha
    val grad = (error.t * training).t + l2_grad
    (loss, grad.map(x => x / m))
  }
}


class SquareLossWithL1L2(override val alpha:Double = 0.5,
                       override val l1_ratio:Double=0.5) extends LossFunc {

  override def costNgradient(training: BDM[Double], label: BDV[Double], weights: BDV[Double]): (Double, BDV[Double]) = {
    val raw_output = training * weights
    val error = raw_output - label
    val m = label.length
    val l2_loss = (1 - l1_ratio) * sum(weights.map(x => x * x))
    val l1_loss = l1_ratio * sum(weights.map(math.abs(_)))
    val loss: Double = error.t * error / (2 * m) + alpha*(l1_loss+l2_loss)
    val l2_grad = weights :* alpha
    val l1_grad = weights.map(sign) :* alpha
    val grad = (error.t * training).t + (l1_grad + l2_grad)
    (loss, grad.map(_/m))
  }
}




