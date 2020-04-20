package com.strings.model.ensemble

/**
 * xgboost 的实现参考 https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/xgboost.py
 *
 * */

import com.strings.model.tree.{CalcInfoGain, DecisionNode}
import com.strings.utils.Utils

import scala.collection.mutable

class TaylorGain(val loss:Loss){
  def _gain(y:Array[Double], y_pred:Array[Double]):Double = {
    val grad = loss.gradient(y,y_pred)
    val nominator = math.pow(y.zip(grad).map(x=> x._1 * x._2).sum,2)
    val denominator = loss.hess(y, y_pred).sum
    0.5 * (nominator / denominator)
  }

  def _gain_by_taylor(y: Array[Array[Double]], y1: Array[Array[Double]], y2: Array[Array[Double]]): Double = {
    val (yy, y_pred) = Utils.split(y)
    val (yy1, y1_pred) = Utils.split(y1)
    val (yy2, y2_pred) = Utils.split(y2)

    val true_gain = _gain(yy1, y1_pred)
    val false_gain = _gain(yy2, y2_pred)
    val gain = _gain(yy, y_pred)
    true_gain + false_gain - gain
  }
}

class ApproximateUpdate(val loss:Loss){
  def leafCalc(y: Array[Array[Double]]): Double = {
    val (yy,y_pred) = Utils.split(y)
    val gradient = loss.gradient(yy,y_pred).zip(yy).map(x => x._1 * x._2).sum
    val hessian = loss.hess(yy,y_pred).sum
    gradient / hessian
  }
}

class XGBoostRegressionTree(val loss:Loss = SquareLoss,
                            val min_samples_split:Int=2,
                            val min_impurity:Double=1e-7,
                            val max_depth:Int = 5) {
  var root: DecisionNode = null
  var catColumns: Set[Int] = Set[Int]()
  var impurity_calculation:TaylorGain = _

  def fit(data: Array[(Array[Double], Array[Double])]): Unit = {
    impurity_calculation = new TaylorGain(loss)
    root = buildtree(data)
  }

  def fit(X: Array[Array[Double]], y: Array[Array[Double]]):Unit={
    val data = y.zip(X)
    fit(data)
  }

  private def buildtree(data: Array[(Array[Double], Array[Double])], current_depth: Int = 0): DecisionNode = {
    var bestGain: Double = 0
    var bestColumn: Int = 0
    var bestValue: Double = 0
    var bestTrueData = Array[(Array[Double], Array[Double])]()
    var bestFalseData = Array[(Array[Double], Array[Double])]()
    val columnSize: Int = data.head._2.length
    val nSamples: Int = data.length
    if (nSamples >= min_samples_split && current_depth <= max_depth) {
      for (col <- 0 until columnSize) {
        var valueSet: Set[Double] = Set()
        for (d <- data) valueSet += d._2(col)
        for (value <- valueSet) {
          val (tData, fData) = data.partition { d =>
            if (catColumns.contains(col)) d._2(col) == value
            else d._2(col) >= value
          }
          val gain = impurity_calculation._gain_by_taylor(data.map(_._1), tData.map(_._1), fData.map(_._1))
          if (gain > bestGain && tData.length > 0 && fData.length > 0) {
            bestGain = gain
            bestColumn = col
            bestValue = value
            bestTrueData = tData
            bestFalseData = fData
          }
        }
      }
    }
    if (bestGain > min_impurity) {
      val tnode: DecisionNode = buildtree(bestTrueData, current_depth + 1)
      val fnode: DecisionNode = buildtree(bestFalseData, current_depth + 1)
      new DecisionNode(bestColumn, bestValue, -1, tnode, fnode)
    } else {
      val leafValue = new ApproximateUpdate(loss).leafCalc(data.map(_._1))
      new DecisionNode(-1, -1, leafValue, null, null)
    }
  }

  def predict(x: Array[Array[Double]]): Array[Double] = x.map(xi => root.predict(xi))
  def predict(x: Array[Double]): Double = root.predict(x)

}
