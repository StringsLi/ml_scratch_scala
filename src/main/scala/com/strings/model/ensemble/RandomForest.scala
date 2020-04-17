package com.strings.model.ensemble

import com.strings.model.tree.ClassificationTree

class RandomForest(val nEstimator:Int = 5,
              val min_samples_split:Int = 2,
              val minGain:Double = 0.0,
              val maxDepth:Int = 5,
              val colsSample:String = "sqrt") {

  private val trees: Array[ClassificationTree] = new Array[ClassificationTree](nEstimator)

  def getIndex(x:Array[Double],index:Array[Int]):Array[Double] = {
    val res:Array[Double] = new Array[Double](index.size)
    for((value,index) <- index.zipWithIndex){
      res(index) = x.apply(value)
    }
    res
  }

  def fit(data:Array[(Double,Array[Double])]): Unit ={
    val nFeaturs = data.map(_._2).apply(0).size

    val maxFeatures:Int = colsSample match{
      case "sqrt" =>math.sqrt(nFeaturs).toInt
      case "log2" => math.log(nFeaturs).toInt
      case _ => nFeaturs
    }

    for(i <- 0 until nEstimator){
      val samples = scala.util.Random.shuffle(data.toList).take(data.size/2).toArray
      val index = scala.util.Random.shuffle(Range(0,nFeaturs).toList).take(maxFeatures).toArray
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
    predict.map((_,1)).groupBy(_._1).maxBy(_._2.size)._1
  }

  def predict(feature:Array[Array[Double]]):Array[Double] = {
    feature.map(predict(_))
  }
}

object RandomForest{
  def main(args: Array[String]): Unit = {
    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray

    val data = dataS.map(x => (x.apply(4),x.slice(0,4)))
    //        data.foreach(x => println(x._2.mkString("-")))
    val dtree = new RandomForest(min_samples_split = 4,maxDepth = 5)
    dtree.fit(data)

//    val dotS = dtree.dot(dtree.root)
//    println(dotS)

//    println(dtree.root.toString)
    val pred =  dtree.predict(data.map(_._2)).zip(data.map(_._1)).map(x => if(x._1 == x._2) 1 else 0 )
    println("准确率为: "+pred.sum.toDouble / data.size)
  }
}
