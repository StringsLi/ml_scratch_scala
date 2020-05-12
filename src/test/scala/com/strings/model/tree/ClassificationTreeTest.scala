package com.strings.model.tree

import com.strings.data.Data
import com.strings.model.metric.Metric

object ClassificationTreeTest {

  def main(args: Array[String]): Unit = {
    val data = Data.iris4MutilClassification()
    val train_test_data = Data.train_test_split(data._1,data._2,0.2,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    val clf = new ClassificationTree()
    clf.fit(trainX,trainY)
    val pred = clf.predict(testX)
    val acc =  Metric.accuracy(pred,testY) * 100
    println(f"准确率为: $acc%-5.2f%%")

    val dotGraph = clf.dot(clf.root)
    println(dotGraph)
    println(clf.root.toString)
  }

}
