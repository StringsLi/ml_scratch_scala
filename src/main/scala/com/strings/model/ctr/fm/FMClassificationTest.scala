package com.strings.model.ctr.fm

import breeze.linalg.{DenseMatrix, DenseVector}
import com.strings.data.Data
import com.strings.model.metric.Metric

object FMClassificationTest {

  def main(args: Array[String]): Unit = {

    val data = Data.iris4BinaryClassification()
    val train_test_data = Data.train_test_split(data._1,data._2,0.4,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    /***
     * 程序不太健壮，有时候的结果特别不好。
     */
    val fm_clf = new FMClassification(learning_rate = 0.05,max_iter = 10)
    fm_clf.fit(DenseMatrix(trainX:_*),DenseVector(trainY:_*))
    val pred = fm_clf.predict(DenseMatrix(testX:_*))
    val acc =  Metric.accuracy(pred.toArray,testY) * 100
    println(f"准确率为: $acc%-5.2f%%")


  }

}
