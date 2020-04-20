package com.strings.model.linear

import breeze.linalg.{DenseMatrix, DenseVector}
import com.strings.loss.LogisticLoss

class LogisticRegression(override val lr:Double = 0.1,
                         override val tolerance:Double = 1e-6,
                         override val max_iters:Int = 100) extends BaseRegression {

  override def init_cost() = {
    cost_func = new LogisticLoss
  }

  def sigmoid(inX: Double) = {
    1.0 / (1 + scala.math.exp(-inX))
  }

  override def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    super.predict(x).map(sigmoid(_))
  }

  def predictClass(x: DenseMatrix[Double]): DenseVector[Double] = {
    val output = predict(x)
    output.map(x => if(x > 0.5) 1.0 else 0.0)
  }

}
