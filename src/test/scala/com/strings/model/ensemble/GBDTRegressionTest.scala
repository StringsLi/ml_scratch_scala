package com.strings.model.ensemble

import com.strings.data.Data
import com.strings.model.metric.Metric

object GBDTRegressionTest {

  def main(args: Array[String]): Unit = {
    val data = Data.iris4Regression()
    val train_test_data = Data.train_test_split(data._1,data._2,0.3,seed = 124L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    val reg = new GBDTRegression(nEstimators = 10)
    reg.fit(trainX,trainY)
    val pred = reg.predict(testX)
    pred.zip(testY).foreach(println)

    val acc =  Metric.rmse(pred,testY)
    println(f"rmse: $acc")
  }

}
