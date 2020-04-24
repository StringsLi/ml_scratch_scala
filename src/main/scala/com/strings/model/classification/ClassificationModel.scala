package com.strings.model.classification

import breeze.linalg.{DenseMatrix, DenseVector}
import com.strings.model.Model

abstract class ClassificationModel extends Model{

  override def predict(x: DenseMatrix[Double]): DenseVector[Double]

}
