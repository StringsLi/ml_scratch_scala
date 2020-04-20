package com.strings.model.linear

import breeze.linalg.DenseMatrix
import com.strings.data.Data

object LogisticRegressionTest {

  def main(args: Array[String]): Unit = {
    val data = DenseMatrix(Data.irisData:_*)

    val features = data(0 to 98, 0 to 3)
    val labels = data(0 to 98, 4)

    val model = new LogisticRegression()
    model.fit(features,labels)
    val pred = model.predictClass(features)
    val predAndlabel = pred.toArray.zip(labels.toArray)
    val rate = predAndlabel.filter(f => f._1==f._2).length.toDouble/predAndlabel.length.toDouble
    println("正确率为：" + rate)
  }

}
