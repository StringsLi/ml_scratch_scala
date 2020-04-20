package com.strings.utils

object Utils {

  def log2(x: Double) = Math.log(x) / Math.log(2)

  //Calculate the entropy of label array y
  def calculate_entropy(y:Array[Double]) ={
    val unique_labels = y.toSet.toArray
    var entropy:Double = 0.0
    for (label <-  unique_labels) {
      val count:Int = y.filter(_==label).size
      val p:Double = count / y.size.toDouble
      entropy += -p * log2(p)
    }
    entropy
  }

  def calculate_variance(y:Array[Double]):Double = {
    val size = y.size
    val mean = y.sum / size
    y.map(x => (x - mean) * (x - mean)).sum / size
  }

  def clip(value: Double) = if (value < 0.00001) 0.00001
  else if (value > 0.99999) 0.99999
  else value

  def one_hot(y: Array[Double]): Array[Array[Double]] = {
    val n_samples = y.length
    val n_classes = y.toSet.size
    val one_hot = Array.ofDim[Double](n_samples, n_classes)
    for (i <- 0 to n_samples - 1) {
      one_hot(i)(y(i).toInt) = 1.0
    }
    one_hot
  }

  def split(y:Array[Array[Double]]):(Array[Double],Array[Double]) = {
    (y.map(_(0)),y.map(_(1)))
  }


}
