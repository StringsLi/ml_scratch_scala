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

object LogisticRegression{

  def main(args: Array[String]): Unit = {
    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray
    val data = DenseMatrix(dataS:_*)

    val features = data(0 to 98, 0 to 3)
    val labels = data(0 to 98, 4)

    val model = new LogisticRegression()
    model.fit(features,labels)
    val predictions = model.predictClass(features)
//    println(model.predict(features))
    val predictionsNlabels = predictions.toArray.zip(labels.toArray)
    val rate = predictionsNlabels.filter(f => f._1==f._2).length.toDouble/predictionsNlabels.length.toDouble
    println("正确率为：" + rate)
  }

}
