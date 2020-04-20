package com.strings.model.ensemble

import com.strings.model.tree.RegressionTree
import scala.collection.mutable.ArrayBuffer

abstract class GBDT(val nEstimators:Int = 10,
                    val lr:Double = 0.1,
                    val minSampleSplit:Int = 5,
                    val minImpurity:Double = 1e-5,
                    val maxDepth:Int = 5,
                    val isRegression:Boolean = true) {

  var loss:Loss = _
  def initLoss()

  private val trees: ArrayBuffer[RegressionTree] = new ArrayBuffer[RegressionTree](nEstimators)

  def fit(X:Array[Array[Double]],y:Array[Double]):Unit={
    initLoss
    for(_ <- 0 until nEstimators){
      val tree = new RegressionTree(minSampleSplit,minImpurity,maxDepth)
      trees.append(tree)
    }
    //让第一棵树去拟合模型, 利用最速下降的近似方法，其关键是利用损失函数的负梯度作为提升树算法中的残差的近似值。
    trees(0).fit(X,y)
    val y_pred = trees(0).predict(X)
    for(i <- 1 until nEstimators){
      val gradient = loss.gradient(y,y_pred)
      trees(i).fit(X,gradient)
    }
  }

  def predict(X:Array[Array[Double]]):Array[Double] = {
    val y_pred = trees(0).predict(X)
    for(i <- 1 until trees.length) {
      val pred_i = trees(i).predict(X).map(_*lr)
      for(j <- 0 until y_pred.length){
        y_pred(j) -= pred_i(j)
      }
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

/**
 *  GBDT 只实现了二分类，可以通过一对一实现多分类
 * @param nEstimators 决策树个数
 * @param lr 学习步长
 * @param minSampleSplit 最小分割样本数
 * @param minImpurity 最小纯度
 * @param maxDepth 最大深度
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
