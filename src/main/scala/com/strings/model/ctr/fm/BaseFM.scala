package com.strings.model.ctr.fm

import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.{pow, sigmoid}
import breeze.stats.distributions.Rand
import breeze.stats.mean

abstract class BaseFM(val n_components:Int = 2,
                      val max_iter:Int = 100,
                      val learning_rate:Double = 0.01,
                      val reg_v:Double = 0.1,
                      val reg_w:Double = 0.5,
                      val reg_w0:Double = 0.0) {

  var fmloss: FmLoss = _
  def initLoss():Unit

  var n_features:Int = _
  var n_samples:Int = _
  var w_0:Double = _
  var w:DenseVector[Double] = _
  var v:DenseMatrix[Double] = _

  def fit(X:DenseMatrix[Double],y:DenseVector[Double]):Unit = {
    initLoss()
    n_features = X.cols
    n_samples = X.rows
    w_0 = 0.0
    w = DenseVector.zeros(n_features)
    v = DenseMatrix.rand(n_features,n_components,Rand.gaussian(0.0,0.1))
    _train(X,y)
  }

  def _train(X:DenseMatrix[Double],y:DenseVector[Double]):Unit = {
    for(_ <- 0 until max_iter){
      val y_pred = _predict(X)
      val loss = fmloss.grad(y,y_pred)
      val w_grad  = (X.t * loss):/ n_samples.toDouble
      w_0 -= learning_rate * (mean(loss) + 2 * reg_w0 * w_0)
      w -= learning_rate * w_grad + 2 * reg_w * w

      val x_arr = (0 until X.rows).map(X(_,::))  // 不能使用X(*,::).toIndexedSeq
      for( (sample,ix) <- x_arr.zipWithIndex){
        for (i <- 0 until n_features){
          val part1 = (sample * v).t * sample(i)
          val v_grad = loss(ix) * (v(i,::).t * pow(sample(i),2)).map(part1(0) - _)
          v(i,::).t -= learning_rate * v_grad :+ (2 * reg_v * v(i,::).t)
        }
      }
    }
  }

  def _predict(X:DenseMatrix[Double]):DenseVector[Double]= {
    val linear_output = X * w
    val factors_output = sum(pow(X * v,2) :- pow(X,2) * pow(v,2),Axis._1) :/ 2.0
    factors_output :+ linear_output :+ w_0
  }

  def predict(X:DenseMatrix[Double]):DenseVector[Double]= {
    _predict(X)
  }

}

class FMClassification(override val n_components:Int = 2,
                       override val max_iter:Int = 100,
                       override val learning_rate:Double = 0.01,
                       override val reg_v:Double = 0.1,
                       override val reg_w:Double = 0.5,
                       override val reg_w0:Double = 0.0) extends BaseFM{
  override def initLoss(): Unit = {
    fmloss = BinaryCrossentropyLoss
  }

  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val predictions = super.predict(X)
    val y_pred = sigmoid(predictions)
    y_pred.map(x => if(x >= 0.5) 1.0 else 0.0)
  }
}

class FMRegression(override val n_components:Int = 2,
                       override val max_iter:Int = 100,
                       override val learning_rate:Double = 0.01,
                       override val reg_v:Double = 0.1,
                       override val reg_w:Double = 0.5,
                       override val reg_w0:Double = 0.0) extends BaseFM{
  override def initLoss(): Unit = {
    fmloss = MeanSquaredLoss
  }

  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    super.predict(X)
  }
}

