package com.strings.model.gradient

import breeze.linalg.{max, DenseMatrix => BDM, DenseVector => BDV}
import scala.collection.mutable.ArrayBuffer
import breeze.numerics.abs
import com.strings.data.Dataset

class AdaMax(val epsilon:Double = 1e-6,
             val beta_1:Double = 0.9,
             val beta_2:Double = 0.999,
             val bacthSize:Int = 10,
             override val lr: Double = 0.01,
             override val tolerance: Double = 1e-6,
             override val num_iters: Int = 1000) extends BaseGradient{

  override def fit(x: BDM[Double], y_train: BDV[Double]):(BDV[Double],Int)  = {
    val ones: BDM[Double] = BDM.ones[Double](x.rows, 1)
    val x_train: BDM[Double] = BDM.horzcat(ones, x)
    val n_features: Int = x_train.cols
    var weights: BDV[Double] = BDV.ones[Double](n_features) :* .01 // 注意是:*
    var n_iters: Int = 0
    var m_t = BDV.zeros[Double](n_features)
    var u_t = BDV.zeros[Double](n_features)
    val loss_lst: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    loss_lst.append(0.0)
    val dataset = new Dataset(x_train,y_train)
    var flag = true
    for (_ <- 0 to num_iters if flag) {
      n_iters = n_iters + 1
      val errors: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      val batch_data = dataset.get_minibatch(bacthSize)
      for((mini_X,mini_y) <- batch_data) {
        val error= mini_X * weights - mini_y
        val mini_gradient = (mini_X.t * error) / bacthSize.toDouble
        errors.append(error.t * error)
        m_t = beta_1 * m_t + (1 - beta_1) * mini_gradient
        val m_t_hat = m_t / (1- scala.math.pow(beta_1,n_iters))
        u_t = max(beta_2*u_t,abs(mini_gradient))
        val gradient_adamax =  m_t_hat :/ u_t
        weights -= gradient_adamax * lr //不能写成 :*
      }
      val loss: Double = errors.sum / (2 * bacthSize)
      val delta_loss = loss - loss_lst.apply(loss_lst.size - 1)
      loss_lst.append(loss)
      if (scala.math.abs(delta_loss) < tolerance) {
        flag = false
      }
    }
    (weights, n_iters)
  }

}


