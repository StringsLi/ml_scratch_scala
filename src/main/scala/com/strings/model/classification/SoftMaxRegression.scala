package com.strings.model.classification

import breeze.linalg.{*, Axis, argmax, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.exp
import com.strings.data.Data
import com.strings.model.metric.Metric


class SoftMaxRegression(var lr: Double = 0.01,
                        var tolerance: Double = 1e-6,
                        var num_iters: Int = 1000) extends ClassificationModel {

  var weights: BDM[Double] = _

  def fit(x: BDM[Double], y: BDV[Double]):Unit= {
    val ones = BDM.ones[Double](x.rows, 1)
    val x_train = BDM.horzcat(ones, x)

    val n_cols = x_train.cols
    val n_samples = x_train.rows
    val n_classes = y.toArray.distinct.length
    weights = BDM.ones[Double](n_cols, n_classes) :* 1.0 / n_classes

    for (_ <- 0 to num_iters) {
      val logit = x_train * weights
      val probs = softmax(logit)
      val y_one_hot = one_hot(y)
      val error: BDM[Double] = probs - y_one_hot
      val gradients = (x_train.t * error) :/ n_samples.toDouble
      weights -= gradients :* lr
    }
  }

  def softmax(logits: BDM[Double]): BDM[Double] = {
    val scores = exp(logits)
    val divisor = sum(scores(*, ::))
     scores(::, *).map(_ :/ divisor)
  }

  def one_hot(y: BDV[Double]): BDM[Double] = {
    val n_samples = y.length
    val n_classes = y.toArray.toSet.size
    val one_hot = Array.ofDim[Double](n_samples, n_classes)
    for (i <- 0 until n_samples) {
      one_hot(i)(y(i).toInt) = 1.0
    }
    BDM(one_hot: _*)
  }

  def predict(x: BDM[Double]): BDV[Double] = {
    val ones = BDM.ones[Double](x.rows, 1)
    val x_test = BDM.horzcat(ones, x)
    val predictions = argmax(x_test * weights, Axis._1)
    predictions.map(_.toDouble)
  }

}

object SoftMaxRegression{
  def main(args: Array[String]): Unit = {
    val data = Data.iris4MutilClassification()
    val train_test_data = Data.train_test_split(data._1,data._2,0.3,seed = 1224L)
    val trainX:Array[Array[Double]] = train_test_data._1
    val trainY:Array[Double] = train_test_data._2
    val testX:Array[Array[Double]] = train_test_data._3
    val testY:Array[Double] = train_test_data._4

    val soft = new SoftMaxRegression()
    soft.fit(BDM(trainX:_*),BDV(trainY:_*))
    val pred = soft.predict(BDM(testX:_*))
    val acc =  Metric.accuracy(pred.toArray,testY) * 100
    println(f"准确率为: $acc%-5.2f%%")
  }
}
