package com.strings.model.tree

class DecisionNode(val featureIndex: Int,
                   val threshold: Double,
                   val value:Double,
                   val tnode: DecisionNode,
                   val fnode: DecisionNode){
  def predict(x: Array[Double]): Double = {
    if(tnode != null && fnode != null) {
      if(x(featureIndex) > threshold) tnode.predict(x)
      else fnode.predict(x)
    } else value
  }
  override def toString: String = {
    if(tnode != null && fnode != null) {
      s"col[$featureIndex]" + " <= " + threshold +
        s" ?($fnode): ($tnode)"
    } else s"class[$value]"
  }
}