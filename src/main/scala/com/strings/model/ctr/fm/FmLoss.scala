package com.strings.model.ctr.fm

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.{log, sigmoid}
import breeze.stats.mean
import com.strings.utils.Utils

trait FmLoss {
  def loss(label:DenseVector[Double],pred:DenseVector[Double]):Double
  def grad(label:DenseVector[Double],pred:DenseVector[Double]):DenseVector[Double]
}

object MeanSquaredLoss extends FmLoss{
  override def loss(label: DenseVector[Double], pred: DenseVector[Double]): Double = {
    mean((label :- pred).map(x => x * x))
  }

  override def grad(label: DenseVector[Double], pred: DenseVector[Double]): DenseVector[Double] = {
    -(label :- pred)
  }
}

object BinaryCrossentropyLoss extends FmLoss{

  override def loss(label: DenseVector[Double], pred: DenseVector[Double]): Double = {
    val predict:DenseVector[Double] = pred.map(Utils.clip)
    val res1:DenseVector[Double] = label :*  log(predict)
    val res2 = label.map(1 - _) :* log(predict.map(1 - _))
//    mean(res1 + res2)
    0.0
  }

  override def grad(label: DenseVector[Double], pred: DenseVector[Double]): DenseVector[Double] = {
    val predict1 = pred.map(Utils.clip)
    val predict = sigmoid(predict1)
    predict :- label
  }
}
