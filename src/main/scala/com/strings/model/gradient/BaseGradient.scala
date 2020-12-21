package com.strings.model.gradient


import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

abstract class BaseGradient(val lr: Double = 0.01,
                           val tolerance: Double = 1e-6,
                           val num_iters: Int = 1000) {

  def fit(x: BDM[Double], y_train: BDV[Double]):(BDV[Double],Int)
}
