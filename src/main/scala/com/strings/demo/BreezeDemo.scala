package com.strings.demo

import breeze.linalg.{*, DenseMatrix, DenseVector, max}

object BreezeDemo {

  def main(args: Array[String]): Unit = {

      val denseVec = DenseVector(1.0, 2.0, 3.0, 4.0)
      val denseVec1 =  DenseVector(2.0, 1.0, 5.0, 4.0)
      //  println(denseVec)

    val denseMatrix1 = DenseMatrix(denseVec,denseVec1)
//    println(denseMatrix1.cols)
    println((denseMatrix1 * 3.0))
    println((denseMatrix1 :* 3.0))

    val ss = denseMatrix1(*,::).map(x => x - denseVec)

    println(denseVec.toDenseMatrix)

//    println(denseMatrix1 :- denseVec.toDenseMatrix)denseVec

//      println(max(denseVec,denseVec1))
      val lr = 0.1
      val res1 = lr * denseVec
      val res2 = denseVec :* lr
      val res3 = denseVec :* denseVec
      val res4 = denseVec.t * denseVec

        println("Without * :" + res1)
        println("With * :" + res2)
      //  println(res3)
      //  println(res4)
  }

}
