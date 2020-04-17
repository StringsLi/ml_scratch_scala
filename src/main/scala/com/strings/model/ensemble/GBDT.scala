package com.strings.model.ensemble

import com.strings.model.tree.RegressionTree

abstract class GBDT(val nEstimators:Int = 10,
                    val lr:Double = 0.1,
                    val minSampleSplit:Int = 5,
                    val minImpurity:Double = 1e-5,
                    val maxDepth:Int = 5,
                    val isRegression:Boolean = true) {

  var loss:Loss = null
  def initLoss:Unit

  private val trees: Array[RegressionTree] = new Array[RegressionTree](nEstimators)

  def fit(X:Array[Array[Double]],y:Array[Double]):Unit={
    initLoss
    for(i <- 0 until nEstimators){
      trees(i) = new RegressionTree(minSampleSplit,minImpurity,maxDepth)
    }
    //让第一棵树去拟合模型, 利用最速下降的近似方法，其关键是利用损失函数的负梯度作为提升树算法中的残差的近似值。
    trees(0).fit(X,y)
    val y_pred = trees(0).predict(X)
//    var score = y_pred.clone()
    for(i <- 1 until nEstimators){
      val gradient = loss.gradient(y,y_pred)
      trees(i).fit(X,gradient)
      val pred_i = trees(i).predict(X).map(_*lr)
      for(i <- 0 until y_pred.size){
        y_pred(i) -= pred_i(i)
      }
    }
  }

  def predict(X:Array[Array[Double]]):Array[Double] = {
    val y_pred = trees(0).predict(X)
    for(i <- 1 until nEstimators) {
      val pred_i = trees(i).predict(X).map(_*lr)
      y_pred(i) -= pred_i(i)
    }
    loss.transform(y_pred)
  }

}

class GBDTRegression(override val nEstimators:Int = 10,
                     override val lr:Double = 0.1,
                     override val minSampleSplit:Int = 5,
                     override val minImpurity:Double = 1e-6,
                     override val maxDepth:Int = 5) extends GBDT{

  override def initLoss: Unit = {
    loss = SquareLoss
  }

}

//object GBDTRegression{
//  def main(args: Array[String]): Unit = {
//    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
//      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
//      .toArray
//    val data = dataS.map(x => (x.apply(3),x.slice(0,3)))
//    val dtree = new GBDTRegression
//    dtree.fit(data.map(_._2),data.map(_._1))
//    val predNactu = dtree.predict(data.map(_._2)).zip(data.map(_._1))
//    predNactu.foreach(println)
//    val acc = predNactu.map(x => (x._1 - x._2)/x._2 )
//    println("准确率为: "+ acc.sum / data.size)
//  }
//}

/**
 *  GBDT 只实现了二分类，可以通过一对一实现多分类
 * @param nEstimators
 * @param lr
 * @param minSampleSplit
 * @param minImpurity
 * @param maxDepth
 */
class GBDTClassification(override val nEstimators:Int = 10,
                         override val lr:Double = 0.1,
                         override val minSampleSplit:Int = 5,
                         override val minImpurity:Double = 1e-6,
                         override val maxDepth:Int = 5) extends GBDT{
  override def initLoss: Unit = {
    loss = LogisticLoss
  }

  override def fit(X: Array[Array[Double]], y: Array[Double]): Unit = {
    super.fit(X, y)
  }

  override def predict(X: Array[Array[Double]]): Array[Double] = {
    super.predict(X).map(x => if(x > 0.5) 1.0 else 0.0)
  }
}

object GBDTClassification{
  def main(args: Array[String]): Unit = {
    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray

    val data = dataS.map(x => (x.apply(4),x.slice(0,4))).slice(0,99)

    val dtree = new GBDTClassification()
    dtree.fit(data.map(_._2),data.map(_._1))

    val predNy =  dtree.predict(data.map(_._2)).zip(data.map(_._1))

    predNy.foreach(println)

    val acc = predNy.map(x => if(x._1 == x._2) 1 else 0 )
    println("准确率为: "+acc.sum.toDouble / data.size)
  }
}
