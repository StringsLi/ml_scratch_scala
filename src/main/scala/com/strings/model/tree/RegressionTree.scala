package com.strings.model.tree

class RegressionTree(override val min_samples_split:Int=2,
                         override val min_impurity:Double=1e-7,
                         override val max_depth:Int = 5) extends DecisionTree {
  override def init_impurity_calc(): Unit = {
    _impurity_calculation = VarianceCalcGain
  }
  override def init_leaf_value_calc(): Unit = {
    _leaf_value_calc = MeanCalc
  }
}
