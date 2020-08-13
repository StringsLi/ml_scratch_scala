package com.strings.model.network

import breeze.linalg.{*, DenseMatrix, max}
import breeze.numerics.exp

trait ActivationFunc {

  def value(x:DenseMatrix[Double]):DenseMatrix[Double]

  def gradient(x:DenseMatrix[Double]):DenseMatrix[Double]

}

object Relu extends ActivationFunc{

  def relu_grad(value: Double):Double = {if (value <= 0) {0} else {1}}
  override def value(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    x.map(i => max(0.0, i))
  }

  override def gradient(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    x.map(relu_grad)
  }

}

object SoftMax extends ActivationFunc{
  override def value(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val softmax = exp(x)
    val divisor = breeze.linalg.sum(softmax(*, ::))
    for (i <- 0 until softmax.cols){
      softmax(::, i) := softmax(::, i) :/ divisor
    }
    softmax
  }

  override def gradient(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val p:DenseMatrix[Double] = value(x)
    p :* p.map(1 - _)
  }
}
