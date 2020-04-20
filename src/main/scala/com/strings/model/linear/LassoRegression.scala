package com.strings.model.linear

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.strings.loss.SquareLossWithL1

class LassoRegression(override val lr:Double = 0.1,
                       override val tolerance:Double = 1e-6,
                       override val max_iters:Int = 500,
                       val alpha:Double = 0.5) extends BaseRegression{

  override def init_cost() = {
    cost_func = new SquareLossWithL1(alpha = alpha)
  }
  override def predict(x: BDM[Double]): BDV[Double] = {
    super.predict(x)
  }
}
