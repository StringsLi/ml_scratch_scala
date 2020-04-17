package com.strings.model.linear

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.strings.loss.LossFunc
import com.strings.model.Model
import scala.collection.mutable.ArrayBuffer

abstract class BaseRegression(val lr:Double = 0.1,
                     val tolerance:Double = 1e-8,
                     val max_iters:Int = 10000) extends Model {

  var errors:ArrayBuffer[Double] = new ArrayBuffer[Double]()
  var weights:BDV[Double] = _
  var n_samples: Int = _
  var n_features: Int = _
  var cost_func:LossFunc = _


  override def predict(x: BDM[Double]): BDV[Double] = {
     val x_test = _add_intercept(x)
     val output = x_test * weights
     output
  }

  def init_cost():Unit

  def _add_intercept(X:BDM[Double]):BDM[Double] = {
    val ones = BDM.ones[Double](X.rows, 1)
    BDM.horzcat(ones, X)
  }

  def fit(X:BDM[Double],y:BDV[Double]):Unit={
    init_cost()
    n_features = X.cols
    n_samples = X.rows
    val init_weights = BDV.ones[Double](n_features + 1) :* 0.01 // 注意是:*
    val X_train = _add_intercept(X)
    gradient_descent(X_train,y,init_weights)
  }

  def gradient_descent(X:BDM[Double],y:BDV[Double],init_weight:BDV[Double]):BDV[Double] =  {

    val theta = BDV.zeros[Double](init_weight.length)
    for (i <- 0 until theta.length) theta(i) = init_weight(i)

    errors.append(0.0)
    var flag = true
    for (_ <- 0 to max_iters if flag){
      val (cost, grad) = cost_func.costNgradient(X,y,theta)
      val delta_loss = cost - errors.apply(errors.size - 1)
      errors.append(cost)
      if (scala.math.abs(delta_loss) < tolerance) {
        flag = false
      } else {
        theta :-= grad :* lr
      }
    }
    weights = theta
    weights
  }

}
