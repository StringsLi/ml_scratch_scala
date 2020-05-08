package com.strings.model.ctr.fm

import breeze.linalg.{DenseMatrix, DenseVector}
import com.strings.data.Data
import com.strings.model.metric.Metric

object FMRegressionTest {

  def main(args: Array[String]): Unit = {
    val data = Data.iris4Regression()
    val train_test_data = Data.train_test_split(data._1,data._2,0.3,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    val fm_clf = new FMRegression(learning_rate = 0.1,max_iter = 100)
    fm_clf.fit(DenseMatrix(trainX:_*),DenseVector(trainY))
    val pred = fm_clf.predict(DenseMatrix(testX:_*))
    val acc =  Metric.accuracyReg(pred.toArray,testY) * 100

    println(pred)
    println(f"准确率为: $acc%-5.2f%%")

  }

}
