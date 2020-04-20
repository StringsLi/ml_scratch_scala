package com.strings.model.ensemble

import com.strings.model.tree.ClassificationTree

class Bagging(val nEstimator:Int = 5,
              val min_samples_split:Int = 2,
              val minGain:Double = 1e-7,
              val maxDepth:Int = 5){

  private val trees: Array[ClassificationTree] = new Array[ClassificationTree](nEstimator)

  def fit(X:Array[Array[Double]],y:Array[Double]):Unit = {
    val data = y.zip(X)
    fit(data)
  }

  def fit(data:Array[(Double,Array[Double])]): Unit ={
    for(i <- 0 until nEstimator){
      trees(i) = new ClassificationTree(min_samples_split,minGain,maxDepth)
      val samples = scala.util.Random.shuffle(data.toList).take(data.length/2).toArray
      trees(i).fit(samples)
    }
  }

  def predict(feature:Array[Double]):Double = {
    val predict:Array[Double] = trees.map(x => x.predict(feature))
    predict.map((_,1)).groupBy(_._1).maxBy(_._2.length)._1
  }

  def predict(feature:Array[Array[Double]]):Array[Double] = {
    feature.map(predict)
  }
}
