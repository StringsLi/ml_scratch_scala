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

object RegressionTree{
  def main(args: Array[String]): Unit = {
    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray

    val data = dataS.map(x => (x.apply(3),x.slice(0,3)))
    //        data.foreach(x => println(x._2.mkString("-")))
    val dtree = new RegressionTree(min_samples_split = 4,max_depth = 5)
    dtree.fit(data)
//    println(dtree.root.toString)
    val predNactu = dtree.predict(data.map(_._2)).zip(data.map(_._1))
    predNactu.foreach(println)
    val acc = predNactu.map(x => (x._1 - x._2)/x._2 )
    println("准确率为: "+ acc.sum / data.size)


  }
}
