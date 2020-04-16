package com.strings.ignore

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._

class Logit(
                          val training: DenseMatrix[Double],
                          val target: DenseVector[Double]
                        ) {

  def costFuncAndGrad(coef: DenseVector[Double]): (Double, DenseVector[Double]) = {
    val xBeta = training * coef
    val expXBeta = exp(xBeta)
    val cost = -sum((target :* xBeta) - log1p(expXBeta))
    val probs = sigmoid(xBeta)
    val grad = training.t * (probs - target)
    (cost, grad)
  }

  private def calOptimalCoef: DenseVector[Double] = {
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(parameters: DenseVector[Double]) = {
        costFuncAndGrad(parameters)
      }
    }
    minimize(f, DenseVector.zeros[Double](training.cols))
  }

  lazy val optimalCoef = calOptimalCoef
}

object Logit{
  def main(args: Array[String]): Unit = {
    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray
    val data = DenseMatrix(dataS:_*)

    val features = data(0 to 98, 0 to 3)
    val labels = data(0 to 98, 4)

    val regressor = new Logit(features, labels)
    val coef = regressor.optimalCoef
    println("The optimal coefficients for Height Weight Data is:")
    println(coef)
  }
}
