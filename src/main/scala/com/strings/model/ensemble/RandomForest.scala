package com.strings.model.ensemble

import com.strings.model.tree.ClassificationTree

class RandomForest(val nEstimator:Int = 10,
              val min_samples_split:Int = 2,
              val minGain:Double = 1e-7,
              val maxDepth:Int = 5,
              val colsSample:String = "sqrt") {

  private val trees: Array[ClassificationTree] = new Array[ClassificationTree](nEstimator)

  def getIndex(x:Array[Double],index:Array[Int]):Array[Double] = {
    val res:Array[Double] = new Array[Double](index.length)
    for((value,index) <- index.zipWithIndex){
      res(index) = x.apply(value)
    }
    res
  }

  def fit(X:Array[Array[Double]],y:Array[Double]):Unit= {
    val data = y.zip(X)
    fit(data)
  }

  def fit(data:Array[(Double,Array[Double])]): Unit ={
    val nFeatures = data.map(_._2).apply(0).length
    val maxFeatures:Int = colsSample match{
      case "sqrt" =>math.sqrt(nFeatures).toInt
      case "log2" => math.log(nFeatures).toInt
      case _ => nFeatures
    }

    for(i <- 0 until nEstimator){
      val samples = scala.util.Random.shuffle(data.toList).take(data.length/2).toArray
      val index = scala.util.Random.shuffle(Range(0,nFeatures).toList).take(maxFeatures).toArray
      val colsSample = samples.map(x => (x._1,getIndex(x._2,index)))
      trees(i) = new ClassificationTree(min_samples_split,minGain,maxDepth)
      trees(i).featureIndex = index
      trees(i).fit(colsSample)
    }
  }

  def predict(feature:Array[Double]):Double = {
    val predict:Array[Double] = trees.map{tree =>
      val index = tree.featureIndex
      tree.predict(getIndex(feature,index))
    }
    predict.map((_,1)).groupBy(_._1).maxBy(_._2.length)._1
  }

  def predict(feature:Array[Array[Double]]):Array[Double] = {
    feature.map(predict)
  }
}
