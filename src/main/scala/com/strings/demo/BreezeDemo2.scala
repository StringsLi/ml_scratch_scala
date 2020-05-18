package com.strings.demo

import breeze.linalg.{Axis, DenseMatrix, DenseVector, clip, inv, pinv, sum}
import breeze.numerics.pow

object BreezeDemo2 {

  def main1(args: Array[String]): Unit = {

    def _predict(X:DenseMatrix[Double],w:DenseVector[Double],v:DenseMatrix[Double],w_0:Double):DenseVector[Double]= {
      val linear_output = X * w
      val factors_output = sum(pow(X * v,2) :- pow(X,2) * pow(v,2),Axis._1) :/ 2.0
      factors_output :+ linear_output :+ w_0
    }

    val x = Array(Array(1.0,2.0,3.0,4.0),Array(5.0,6.0,7.0,8.0))

    val X = DenseMatrix(x:_*)
    val w = DenseVector(Array(1.0,2.0,3.0,5.0))
    val w_0 = 1.0
    val vv = Array(Array(1.0,2.0),Array(3.0,4.0),Array(5.0,6.0),Array(7.0,8.0))
    val v = DenseMatrix(vv:_*)

    val res = _predict(X,w,v,w_0)

    println(res)

  }

  def main(args: Array[String]): Unit = {

    val sample = DenseVector(Array(1.0,2.0,3.0))
    val mean = DenseVector(Array(1.0,1.0,1.0))

    val arr = Array(Array(1.0,2.0,5.0),Array(4.0,5.0,6.0),Array(7.0,8.0,9.0))

    val covar = DenseMatrix(arr:_*)
    println(pinv(covar)*covar)

    val gram = (sample :- mean).t * pinv(covar) * (sample :- mean)

    println(gram)

    val ff = Array(Array(0.0,1.0),Array(1.0,0.0))
    val Q = DenseMatrix(ff:_*)
    val dd = clip(Q,1.0e-100,0.6)
    println(dd)
  }

}
