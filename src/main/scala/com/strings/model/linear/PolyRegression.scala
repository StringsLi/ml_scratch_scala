package com.strings.model.linear

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.strings.loss.SquareLoss
import scala.collection.mutable.ArrayBuffer

object PolyRegression{

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
      }toList
    }
  }

}

class PolyRegression(val degree:Int = 2) extends BaseRegression {
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
