package com.strings.model.gradient

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.strings.data.Dataset

import scala.collection.mutable.ArrayBuffer


object RMSProp{
  def main(args: Array[String]): Unit = {
    val num_inputs = 2
    val num_examples = 10000
    val x_train: BDM[Double] = BDM.rand(num_examples, num_inputs)
    val ones = BDM.ones[Double](num_examples, 1)
    val x_cat = BDM.horzcat(ones, x_train)
    val nos = BDV.rand(num_examples) * 0.1
    val y_train = x_cat * BDV(2.3, 6.4, -3.2) + nos

    val model = new RMSProp(lr = 0.1,num_iters = 1000)
    val weights = model.fit(x_train, y_train)
    println("梯度下降求解的权重为：" + weights._1)
    println("梯度下降迭代次数：" + weights._2)
  }
}


class RMSProp(val epsilon:Double = 1e-6,
              val gamma:Double = 0.9,
              val bacthSize:Int = 10,
              override val lr: Double = 0.01,
              override val tolerance: Double = 1e-8,
              override val num_iters: Int = 10000) extends BaseGradient {

  override def fit(x: BDM[Double], y_train: BDV[Double]):(BDV[Double],Int)  = {
    val ones: BDM[Double] = BDM.ones[Double](x.rows, 1)
    val x_train: BDM[Double] = BDM.horzcat(ones, x)
    val n_features: Int = x_train.cols
    var weights: BDV[Double] = BDV.ones[Double](n_features) :* .01 // 注意是:*
    var n_iters: Int = 0
    var gradient_exp = BDV.zeros[Double](n_features)
    val loss_lst: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    loss_lst.append(0.0)
    val dataset = new Dataset(x_train,y_train)
    var flag = true
    while (n_iters < num_iters & flag) {
      val errors: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      val batch_data = dataset.get_minibatch(bacthSize)
      for((mini_X,mini_y) <- batch_data) {
        val error= mini_X * weights - mini_y
        val mini_gradient = (mini_X.t * error) / bacthSize.toDouble
        errors.append(error.t * error)
        gradient_exp = (gamma * gradient_exp) + ((1 - gamma) * (mini_gradient :* mini_gradient))
        val gradient_rms =  mini_gradient :/gradient_exp.map(x => scala.math.sqrt(x) + epsilon)
        weights -= gradient_rms * lr //不能写成 :*
      }
      val loss: Double = errors.sum / (2 * bacthSize)
      val delta_loss = loss - loss_lst.apply(loss_lst.size - 1)
      loss_lst.append(loss)
      if (scala.math.abs(delta_loss) < tolerance) {
        flag = false
      }
    }
    n_iters = n_iters + 1
    (weights, n_iters)
  }

}

