package com.strings.model.ensemble

import com.strings.data.Data
import com.strings.model.metric.Metric

object AdaBoostClassificationTest {
    def main(args: Array[String]): Unit = {
      val data = Data.iris4BinaryClassification()

      val y_trans = data._2.map(x => if(x == 0.0) -1.0 else 1.0)

      val train_test_data = Data.train_test_split(data._1,y_trans,0.3,seed = 223L)
      val trainX = train_test_data._1
      val trainY = train_test_data._2
      val testX = train_test_data._3
      val testY = train_test_data._4

      val clf = new AdaBoost(nEstimator = 5 )
      clf.fit(trainX,trainY)
      val pred = clf.predict(testX)
      val acc =  Metric.accuracy(pred,testY) * 100
      println(f"准确率为: $acc%-5.2f%%")
    }

}
