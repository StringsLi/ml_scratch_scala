package com.strings.ignore

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

class Linear {

  def computeCostAndGrad(X: BDM[Double], y: BDV[Double])(theta: BDV[Double]): (Double, BDV[Double]) = {
    val diff = X * theta - y
    val m = y.length
    val cost = (diff.t * diff) / (2 * m)
    val grad:BDV[Double] = (diff.t * X).t
    (cost, grad.map(x => x/m))
  }


  def gradientDescent(initTheta: BDV[Double], func: BDV[Double] => (Double, BDV[Double]), alpha: Double, num_iters: Int): (BDV[Double], BDV[Double]) = {
    println("=== start gradientDescent loop ===")
    //initialize theta
    val theta = BDV.zeros[Double](initTheta.length)
    for (i <- 0 until theta.length) theta(i) = initTheta(i)
    val costHist = BDV.zeros[Double](num_iters)

    for (n <- 0 until num_iters) {
      print((n + 1) + "/" + num_iters + " : ")
      val r = func(theta)
      costHist(n) = r._1
      println("cost = " + r._1 + "  theta = " + theta.toArray.mkString(" "))
      theta :-= (r._2 :* alpha)
    }
    println("=== finish gradientDescent loop ===")
    (theta, costHist)
  }


}

object Linear{
  def main(args: Array[String]): Unit = {

    val num_inputs = 2
    val num_examples = 10000
    val x_train: BDM[Double] = BDM.rand(num_examples, num_inputs)
    val ones = BDM.ones[Double](num_examples, 1)
    val x_cat = BDM.horzcat(ones, x_train)
    val nos = BDV.rand(num_examples) * 0.1
    val y_train = x_cat * BDV(2.8, 6.4, -2.2) + nos

    // learning parameters
    val alpha = 0.1d
    val num_iters = 1000
    val initTheta:BDV[Double] = BDV.ones[Double](3) :* 0.01

    val linear  = new Linear

    // learning
    val (learnedTheta, histOfCost) = linear.gradientDescent(initTheta, linear.computeCostAndGrad(x_cat, y_train), alpha, num_iters)

    println(learnedTheta)
  }
}
