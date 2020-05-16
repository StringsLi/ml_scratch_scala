package com.strings.model.regression

import breeze.linalg.{DenseMatrix, DenseVector}
import com.strings.model.Model

abstract class RegressionModel extends Model{

  override def predict(x: DenseMatrix[Double]): DenseVector[Double]
}
