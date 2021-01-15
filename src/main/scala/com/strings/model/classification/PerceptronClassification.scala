package com.strings.model.classification
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.round
import com.strings.data.Data
import com.strings.model.metric.Metric

class PerceptronClassification(val lr:Double = 0.01,
                               val iterations:Int = 1000) extends ClassificationModel {

  private var weights: DenseVector[Double] = _

  def fit(x:DenseMatrix[Double],y:DenseVector[Double]): Unit ={
    val ones = DenseMatrix.ones[Double](x.rows, 1)
    val xTrain = DenseMatrix.horzcat(ones, x)

    weights = DenseVector.ones[Double](x.cols + 1) :* 0.01

    for(_ <- 0 until iterations){
      val output:DenseVector[Double] = round(xTrain * weights).map(_.toDouble)
      val gradient:DenseVector[Double] = xTrain.t * (y - output)
      weights :+= lr * gradient.map(_/xTrain.rows)
    }

  }

  override def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    val ones = DenseMatrix.ones[Double](x.rows, 1)
    val xTrain = DenseMatrix.horzcat(ones, x)

    val y_pred = xTrain * weights

    round(y_pred).map(_.toDouble)

  }
}

object PerceptronClassification{
  def main(args: Array[String]): Unit = {

    val iris_data = Data.iris4BinaryClassification()
    val train_test_data = Data.train_test_split(iris_data._1,iris_data._2,0.32,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    val nb = new PerceptronClassification()
    nb.fit(DenseMatrix(trainX:_*),DenseVector(trainY))
    val pred1 = nb.predict(DenseMatrix(testX:_*))
    val acc =  Metric.accuracy(pred1.toArray,testY) * 100
    println(pred1)
    println(testY.toList)

    println(f"准确率为: $acc%-5.2f%%")

  }

}
