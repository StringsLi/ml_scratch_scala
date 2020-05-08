package com.strings.demo

import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.pow

object Test3 {

  def main(args: Array[String]): Unit = {

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

}
