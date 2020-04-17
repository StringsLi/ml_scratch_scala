package com.strings.model.ensemble

import com.strings.model.tree.ClassificationTree

class Bagging(val nEstimator:Int = 5,
              val min_samples_split:Int = 2,
              val minGain:Double = 0.0,
              val maxDepth:Int = 5){

  private val trees: Array[ClassificationTree] = new Array[ClassificationTree](nEstimator)

  def fit(data:Array[(Double,Array[Double])]): Unit ={
    for(i <- 0 until nEstimator){
      val samples = scala.util.Random.shuffle(data.toList).take(data.size/2).toArray
      trees(i) = new ClassificationTree(min_samples_split,minGain,maxDepth)
      trees(i).fit(samples)
    }
  }

  def predict(feature:Array[Double]):Double = {
    val predict:Array[Double] = trees.map(x => x.predict(feature))
    predict.map((_,1)).groupBy(_._1).maxBy(_._2.size)._1
  }

  def predict(feature:Array[Array[Double]]):Array[Double] = {
    feature.map(predict(_))
  }
}

object Bagging{
  def main(args: Array[String]): Unit = {
    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray

    val data = dataS.map(x => (x.apply(4),x.slice(0,4)))
    val dtree = new Bagging(min_samples_split = 4,maxDepth = 5)
    dtree.fit(data)

    val pred =  dtree.predict(data.map(_._2)).zip(data.map(_._1)).map(x => if(x._1 == x._2) 1 else 0 )
    println("准确率为: "+pred.sum.toDouble / data.size)
  }
}
