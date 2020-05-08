package com.strings.model.ensemble

import com.strings.utils.Utils
import scala.collection.mutable.ArrayBuffer

class DecisionStump(var featureIndex: Int = -1,
                    var threshold: Double = Double.PositiveInfinity,
                    var polarity:Int = 1,
                    var alpha:Double = 0.0){

  override def toString: String = {
    s"featureIndex:$featureIndex, threshold:$threshold, alpha:$alpha"
  }
}

class AdaBoost(val nEstimator:Int = 5) {
  private val trees: ArrayBuffer[DecisionStump] = new ArrayBuffer[DecisionStump]()
  def fit(X:Array[Array[Double]],y:Array[Double]):Unit = {
    val n_samples = X.length
    val n_features = X(0).length
    var w = Array.fill(n_samples)(1.0 / n_samples)
    for(_ <- 0 until nEstimator){
      val clf = new DecisionStump()
      var min_error = Double.PositiveInfinity
      for(feature_i <- 0 until n_features){
        val feauture_value = X.map(_(feature_i))
        val unique_values = feauture_value.distinct.toList
        for(threshold <- unique_values){
          var p = 1
          val prediction = Array.fill(y.length)(1.0)
          X.map(_(feature_i)).zipWithIndex.filter(_._1 < threshold).foreach{
            case(_,idx) =>
              prediction(idx) = -1.0
          }
          val diff_index = prediction.zip(y).zipWithIndex.filter(x =>  x._1._1 != x._1._2).map(_._2)
          var error = 0.0
          diff_index.foreach{ i =>
            error += w(i)
          }
          if(error > 0.5) {
            error = 1 - error
            p = -1
          }
          if(error < min_error) {
            clf.polarity = p
            clf.threshold = threshold
            clf.featureIndex = feature_i
            min_error = error
          }
        }
      }
      clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
      val predictions = Array.fill(y.length)(1.0)
      val negative_idx = X.map(_(clf.featureIndex)).zipWithIndex.filter(_._1 * clf.polarity < clf.polarity * clf.threshold).map(_._2)
      negative_idx.foreach(predictions(_) = -1.0)

      for(i <- 0 until n_samples){
        w(i) *= math.exp(clf.alpha*y(i)*predictions(i))
      }
      w = w.map(_/w.sum)
      trees.append(clf)
    }
  }
  def predict(X:Array[Array[Double]]):Array[Double] = {
    val n_samples = X.length
    val y_pred = Array.fill(n_samples)(0.0)
    for(clf <- trees){
      val predictions = Array.fill(n_samples)(1.0)
      val negative_idx = X.map(_(clf.featureIndex)).zipWithIndex.filter(_._1 * clf.polarity < clf.polarity * clf.threshold).map(_._2)
      negative_idx.foreach(predictions(_) = -1.0)
      for(i <- 0 until n_samples){
        y_pred(i) += clf.alpha * predictions(i)
      }
    }
    y_pred.map(Utils.sign)
  }
}

