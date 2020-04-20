package com.strings.model.tree

import com.strings.model.ensemble.Loss

abstract class LeafValueCalc(val loss:Loss = null) {
  def leafCalc(y:Array[Double]):Double
}

object MajorityCalc extends LeafValueCalc{
  override def leafCalc(y: Array[Double]): Double = {
    y.map((_,1)).groupBy(_._1).map(x=>(x._1,x._2.size)).maxBy(_._2)._1
  }
}

object MeanCalc extends LeafValueCalc{
  override def leafCalc(y: Array[Double]): Double = {
    y.sum / y.size
  }
}


