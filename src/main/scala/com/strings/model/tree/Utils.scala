package com.strings.model.tree

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


}
