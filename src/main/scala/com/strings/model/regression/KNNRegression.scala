package com.strings.model.regression

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.strings.data.Data
import com.strings.model.metric.Metric

/**
 *
 * @param k the neigbors of the dataset
 * @param dataX featurs
 * @param dataY labels
 * @param distanceFunction distance function type: Euclidean ,Manhattan
 */

class KNNRegression(k: Int,
                        dataX: DenseMatrix[Double],
                        dataY: Seq[Double],
                        distanceFunction: String = "Euclidean") extends RegressionModel {

  private val  distanceFn = distanceFunction match {
    case "Euclidean" => (v1: DenseVector[Double], v2: DenseVector[Double])
    =>  v1.toArray.zip(v2.toArray)
      .map(x => scala.math.pow(x._1 - x._2, 2)).sum

    case "Manhattan" => (v1: DenseVector[Double], v2: DenseVector[Double])
    => v1.toArray.zip(v2.toArray).map(x => scala.math.abs(x._1-x._2)).sum
  }

  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    X(*,::).map(x => predict(x))
  }

  def predict(feature:DenseVector[Double]):Double = {
    val topKClass =  dataX(*,::).map { x =>
      distanceFn(x,feature)
    }.toArray.zipWithIndex.sortBy(_._1).take(k).map{case (_,idx) => dataY(idx)}
    topKClass.foldLeft(0.0)((x1,x2) => x1 + x2) / k.toDouble
  }

}

object KNNClassification{

  def main(args: Array[String]): Unit = {
    val data = Data.iris4Regression()
    val train_test_data = Data.train_test_split(data._1,data._2,0.2,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    val knn = new KNNRegression(k = 7,DenseMatrix(trainX:_*),trainY)
    val pred = knn.predict(DenseMatrix(testX:_*))
    val acc =  Metric.accuracyReg(pred.toArray,testY) * 100
    println(f"准确率为: $acc%-5.2f%%")

  }
}

