package com.strings.data

object Data {

  val irisData = scala.io.Source.fromFile("D:/data/iris.csv")
            .getLines().toSeq.tail
            .map{_.split(",")
            .filter(_.length() > 0)
            .map(_.toDouble)}
            .toArray

  def iris4BinaryClassification():Array[(Double,Array[Double])] = {
    val data = irisData.map(x => (x.apply(4),x.slice(0,4))).slice(0,99)
    data
  }

  def iris4MutilClassification():Array[(Double,Array[Double])] = {
    val data = irisData.map(x => (x.apply(4),x.slice(0,4)))
    data
  }

  def iris4Regression():Array[(Double,Array[Double])] ={
    val data = irisData.map(x => (x.apply(3), x.slice(0, 3)))
    data
  }

}
