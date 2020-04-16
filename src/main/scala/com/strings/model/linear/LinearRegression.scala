package com.strings.model.linear

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.strings.loss.SquareLoss

class LinearRegression(override val lr:Double = 0.1,
                       override val tolerance:Double = 1e-6,
                       override val max_iters:Int = 10 ) extends BaseRegression{

  override def init_cost() = {
    cost_func = new SquareLoss
  }
  override def predict(x: BDM[Double]): BDV[Double] = {
    super.predict(x)
  }

}

object LinearRegression{
  def main(args: Array[String]): Unit = {
    val num_inputs = 2
    val num_examples = 10000
    val x_train: BDM[Double] = BDM.rand(num_examples, num_inputs)
    val ones = BDM.ones[Double](num_examples, 1)
    val x_cat = BDM.horzcat(ones, x_train)
    val nos = BDV.rand(num_examples) * 0.1
    val y_train = x_cat * BDV(2.8, 6.4, -2.2) + nos

    val model = new LinearRegression(lr = 1,max_iters = 100)
    model.fit(x_train, y_train)
    println(model.max_iters)
    println(model.lr)
    println("权重为：" + model.weights)
  }
}


