package com.strings.model.ensemble

import scala.collection.mutable.ArrayBuffer

abstract class XGBoost(val nEstimators:Int = 10,
                       val lr:Double = 0.1,
                       val minSampleSplit:Int = 2,
                       val minImpurity:Double = 1e-6,
                       val maxDepth:Int = 4) {

  var loss:Loss = _
  def initLoss():Unit

  private val trees: ArrayBuffer[XGBoostRegressionTree] = new ArrayBuffer[XGBoostRegressionTree](nEstimators)

  def fit(X:Array[Array[Double]],y:Array[Double]):Unit={
    initLoss
    for(_ <- 0 until nEstimators){
      val tree = new XGBoostRegressionTree(loss,minSampleSplit,minImpurity,maxDepth)
      trees.append(tree)
    }
    val y_pred:Array[Double] = new Array[Double](y.length)
    for(i <- 0 until nEstimators){
      val y_and_pred:Array[Array[Double]] = Array.ofDim(y.length,2)
      for(k <- 0 until y.length){
        y_and_pred(k)(0) = y(k)
        y_and_pred(k)(1) = y_pred(k)
      }
      trees(i).fit(X,y_and_pred)
      val predI = trees(i).predict(X).map(_*lr)
      for(j <- Range(0, y_pred.size)){
        y_pred(j) -= predI(j)
      }
    }
  }

  def predict(X:Array[Array[Double]]):Array[Double] = {
    val y_pred = trees(0).predict(X)
    for(j <- 1 until nEstimators) {
      val pred_i = trees(j).predict(X).map(_*lr)
      for(i <- Range(0, y_pred.length)){
        y_pred(i) -= pred_i(i)
      }
    }
    loss.transform(y_pred)
  }

}

class XGBoostRegression(override val nEstimators:Int = 10,
                        override val lr:Double = 0.1,
                        override val minSampleSplit:Int = 2,
                        override val minImpurity:Double = 1e-6,
                        override val maxDepth:Int  = 5) extends XGBoost {

  override def initLoss(): Unit = {
    loss = SquareLoss
  }
}

/**
 *  Xgboost只实现了二分类，可以通过一对一实现多分类
 */
class XGBoostClassification(override val nEstimators:Int = 10,
                         override val lr:Double = 0.1,
                         override val minSampleSplit:Int = 2,
                         override val minImpurity:Double = 1e-7,
                         override val maxDepth:Int = 4) extends XGBoost {
  override def initLoss(): Unit = {
    loss = LogisticLoss
  }

  override def predict(X: Array[Array[Double]]): Array[Double] = {
    super.predict(X).map(x => if(x > 0.5) 1.0 else 0.0)
  }

}
