package com.strings.demo

import breeze.linalg.{*, DenseMatrix, DenseVector, Transpose}
import breeze.stats.mean

object Breeze2 {
  def main(args: Array[String]): Unit = {
    val array = Array(Array(1.0,2,3),Array(4.0,5,6),Array(7.0,8,9))
    val dist:DenseMatrix[Double] = DenseMatrix(array:_*)
    println(dist)
    val M_r:Transpose[DenseVector[Double]] = mean(dist(::,*))
    //    val M_r1:Transpose[DenseVector[Double]]  = mean(dist, Axis._0)
    val M_c:DenseVector[Double] = mean(dist(*,::))
    val meanDist:Double = mean(dist)

    val B1:DenseMatrix[Double] = dist(::,*).map(x => x - M_c)
    val B = B1(*,::).map(x => x - M_r.t )  :+ meanDist
    println(B)

    println(mean(B(*,::)))
    println(mean(B(::,*)))
  }
}
