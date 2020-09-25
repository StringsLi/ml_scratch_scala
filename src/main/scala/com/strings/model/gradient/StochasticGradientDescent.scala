package com.strings.model.gradient

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class StochasticGradientDescent(override val lr: Double = 0.01,
                                override val tolerance: Double = 1e-6,
                                override val num_iters: Int = 1000) extends BaseGradient{

  override def fit(x: BDM[Double], y_train: BDV[Double]):(BDV[Double],Int) = {
    val ones: BDM[Double] = BDM.ones[Double](x.rows, 1)
    val x_train: BDM[Double] = BDM.horzcat(ones, x)
    val n_samples: Int = x_train.rows
    val n_features: Int = x_train.cols
    var weights: BDV[Double] = BDV.ones[Double](n_features) * .01
    val loss_lst: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    loss_lst.append(0.0)
    var flag = true
    var n_iters: Int = 0
    for (_ <- 0 to num_iters if flag) {
      n_iters = n_iters + 1
      val range = Random.shuffle((0 until n_samples).toList)
      val errors: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      for (j <- 0 until n_samples) {
        val error_j = x_train(range(j), ::) * weights - y_train(range(j))
        val gradient_j = x_train(range(j), ::).t * error_j
        errors.append(error_j * error_j)
        weights = weights - lr * gradient_j
      }
      val loss: Double = errors.sum / (2 * errors.size)
      val delta_loss = loss - loss_lst.apply(loss_lst.size - 1)
      loss_lst.append(loss)
      if (scala.math.abs(delta_loss) < tolerance) {
        flag = false
      }
    }

    (weights, n_iters)
  }

}

