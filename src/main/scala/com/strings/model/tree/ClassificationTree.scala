package com.strings.model.tree

class ClassificationTree(override val min_samples_split:Int=2,
                         override val min_impurity:Double=1e-7,
                         override val max_depth:Int = 5) extends DecisionTree {

  var featureIndex: Array[Int] = _

  override def init_impurity_calc(): Unit = {
    _impurity_calculation = EntropyCalcGini
  }

  override def init_leaf_value_calc(): Unit = {
    _leaf_value_calc = MajorityCalc
  }
}
