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


