package com.strings.model

import breeze.linalg.{DenseMatrix => BDM,DenseVector => BDV}

trait Model extends Serializable {
  /**
   * @param x input matrix
   * @return predict vector value
   */
  def predict(x:BDM[Double]) : BDV[Double]

}