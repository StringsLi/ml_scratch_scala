package com.strings.ignore

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.sigmoid

import scala.collection.mutable.ArrayBuffer

class LogisticRegression( val lr: Double = 0.01,
                           val maxIterations: Int = 1000,
                           val stohasticBatchSize: Int = 100,
                           val optimizerType: String = "GradientDescent",
                           val eps: Double = 1e-6,
                           val threshold: Double = 0.5) {
  private var weights:BDV[Double] = null

  def fit(x: BDM[Double], y_train: BDV[Double]): BDV[Double] = {
    val ones = BDM.ones[Double](x.rows, 1)
    val x_train = BDM.horzcat(ones, x)
    val n_samples = x_train.rows
    val n_features = x_train.cols
    weights = BDV.ones[Double](n_features) :* .01 // 注意是:*

    val loss_lst: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    loss_lst.append(0.0)

    var flag = true
    for (i <- 0 to maxIterations if flag) {
      val raw_output = (x_train * weights).map(sigmoid(_))
      val error = raw_output - y_train
      val loss: Double = error.t * error
      val delta_loss = loss - loss_lst.apply(loss_lst.size - 1)
      loss_lst.append(loss)
      if (scala.math.abs(delta_loss) < eps) {
        flag = false
      } else {
        val gradient = (error.t * x_train) :/ n_samples.toDouble
        weights = weights - (gradient :* lr).t
      }
    }
    weights
  }


  def predict(x: BDM[Double]): BDV[Double] = {
    val x_test = BDM.horzcat(BDM.ones[Double](x.rows, 1), x)
    val output = (x_test * weights).map(sigmoid(_))
    output
  }

  def predictClass(x: BDM[Double]): BDV[Double] = {
    val output = predict(x)
    output.map(x => if(x > 0.5) 1.0 else 0.0)
  }

}

object LogisticRegression{
  def main(args: Array[String]): Unit = {
    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray
    val data = BDM(dataS:_*)

    val features = data(0 to 98, 0 to 3)
    val labels = data(0 to 98, 4)

    val model = new LogisticRegression()
    model.fit(features,labels)
    val predictions = model.predictClass(features)
    println(model.predict(features))
    val predictionsNlabels = predictions.toArray.zip(labels.toArray)
    val rate = predictionsNlabels.filter(f => f._1==f._2).length.toDouble/predictionsNlabels.length.toDouble
    println("正确率为：" + rate)
  }
}
