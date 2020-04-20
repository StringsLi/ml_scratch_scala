package com.strings.model.linear

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

object RegressionTest {

  def main(args: Array[String]): Unit = {
    val num_inputs = 2
    val num_examples = 10000
    val x_train: BDM[Double] = BDM.rand(num_examples, num_inputs)
    val ones = BDM.ones[Double](num_examples, 1)
    val x_cat = BDM.horzcat(ones, x_train)
    val nos = BDV.rand(num_examples) * 0.1
    val y_train = x_cat * BDV(2.8, 6.4, -2.2) + nos

    val regArr:Array[BaseRegression] = new Array[BaseRegression](5)
    regArr(0) = new LinearRegression()
    regArr(1) = new LassoRegression()
    regArr(2) = new RidgeRegression()
    regArr(3) = new ElasticRegression()
    regArr(4) = new PolyRegression()

    for(reg <- regArr){
      reg.fit(x_train, y_train)
      println(reg.getClass.getName + "的权重为：" + reg.weights)
    }

    /**
     * 结果如下：
     * com.strings.model.linear.LinearRegression权重为：DenseVector(2.7106222483759748, 1.8594622470359834, 1.097174412563561)
     * com.strings.model.linear.LassoRegression权重为：DenseVector(2.7106031853614714, 1.859427076363167, 1.0971391121125451)
     * com.strings.model.linear.RidgeRegression权重为：DenseVector(2.710567584587612, 1.859422689743183, 1.0971537032365524)
     * com.strings.model.linear.ElasticRegression权重为：DenseVector(2.7105485218791485, 1.8593875198553513, 1.097118403574378)
     * com.strings.model.linear.PolyRegression权重为：DenseVector(2.8922397117660315, 0.21329382239657957, 0.05571585410203446, -0.06571378265679607, 6.151929859284106, -2.162693247051346)
     */

  }

}
