package com.strings.model.linear

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.strings.loss.SquareLoss
import scala.collection.mutable.ArrayBuffer

object PolyRegression{

  def main(args: Array[String]): Unit = {
    val num_inputs = 2
    val num_examples = 100
    val x_train: BDM[Double] = BDM.rand(num_examples, num_inputs)
    val ones = BDM.ones[Double](num_examples, 1)
    val x_cat = BDM.horzcat(x_train,ones)
    val nos = BDV.rand(num_examples) * 0.1
    val y_train = x_cat * BDV(2.8, 6.4, -2.2) + nos

    val model = new PolyRegression(degree = 2)
    model.fit(x_train, y_train)
    println(model.max_iters)
    println(model.lr)
    println("权重为：" + model.weights)
  }

  def calculateCombinedCombinations(degree: Int, vector: BDV[Double]): List[Double] = {
    if(degree == 0) {
      List()
    } else {
      val partialResult = calculateCombinedCombinations(degree - 1, vector)
      val combinations = calculateCombinations(vector.size, degree)
      val result = combinations map {
        combination =>
          combination.zipWithIndex.map{
            case (exp, idx) => math.pow(vector(idx), exp)
          }.fold(1.0)(_ * _)
      }
      result ::: partialResult
    }
  }

  def calculateCombinations(length: Int, value: Int): List[List[Int]] = {
    if(length == 0) {
      List()
    } else if (length == 1) {
      List(List(value))
    } else {
      value to 0 by -1 flatMap {
        v =>
          calculateCombinations(length - 1, value - v) map {
            v::_
          }
      } toList
    }
  }

}

class PolyRegression(val degree:Int) extends BaseRegression {
  override def init_cost(): Unit = {
    cost_func = new SquareLoss()
  }

  override def fit(X: BDM[Double], y: BDV[Double]): Unit = {
    val lst_poly:ArrayBuffer[BDV[Double]] = new ArrayBuffer[BDV[Double]]()
    (0 until X.rows).foreach { i =>
      val x_i: BDV[Double] = X(i, ::).t
      val lst = PolyRegression.calculateCombinedCombinations(degree,x_i)
      lst_poly.append(BDV(lst:_*))
    }
    val X_poly :BDM[Double]   =  BDV.horzcat(lst_poly:_*).t
    super.fit(X_poly, y)
  }
}
