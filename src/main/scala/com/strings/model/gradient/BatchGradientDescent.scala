package com.strings.model.gradient

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import scala.collection.mutable.ArrayBuffer

class BatchGradientDescent(override val lr: Double = 0.01,
                           override val tolerance: Double = 1e-6,
                           override val num_iters: Int = 1000) extends BaseGradient{

  override def fit(x: BDM[Double], y_train: BDV[Double]):(BDV[Double],Int) = {
    val ones = BDM.ones[Double](x.rows, 1)
    val x_train = BDM.horzcat(ones, x)
    val n_samples = x_train.rows
    val n_features = x_train.cols
    var weights = BDV.ones[Double](n_features) :* .01 // 注意是:*
    var n_iters:Int = 0
    val loss_lst: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    loss_lst.append(0.0)

    var flag = true
    for (_ <- 0 to num_iters if flag) {
      n_iters = n_iters + 1
      val raw_output = x_train * weights
      val error = raw_output - y_train
      val loss: Double = error.t * error
      val delta_loss = loss - loss_lst.apply(loss_lst.size - 1)
      loss_lst.append(loss)
      if (scala.math.abs(delta_loss) < tolerance) {
        flag = false
      } else {
        val gradient = (error.t * x_train) / n_samples.toDouble
        weights = weights - (gradient * lr).t
      }
    }
    (weights,n_iters)
  }

  def predict(weights: BDV[Double], x: BDM[Double]): BDV[Double] = {
    val x_test = BDM.horzcat(BDM.ones[Double](x.rows, 1), x)
    val output = x_test * weights
    output
  }
}

