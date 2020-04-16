package com.strings.model.tree

class ClassificationTree(override val min_samples_split:Int=2,
                         override val min_impurity:Double=1e-7,
                         override val max_depth:Int = 5) extends DecisionTree {
  override def init_impurity_calc(): Unit = {
    _impurity_calculation = EntropyCalcGain
  }

  override def init_leaf_value_calc(): Unit = {
    _leaf_value_calc = MajorityCalc
  }
}

object ClassificationTree{
  def main(args: Array[String]): Unit = {

    val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
      .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
      .toArray

    val data = dataS.map(x => (x.apply(4),x.slice(0,4)))
    //        data.foreach(x => println(x._2.mkString("-")))
    val dtree = new ClassificationTree(min_samples_split = 4,max_depth = 5)
    dtree.fit(data)

    val dotS = dtree.dot(dtree.root)
    println(dotS)

    println(dtree.root.toString)
    val pred =  dtree.predict(data.map(_._2)).zip(data.map(_._1)).map(x => if(x._1 == x._2) 1 else 0 )
    println("准确率为: "+pred.sum.toDouble / data.size)
  }
}
