package com.strings.ignore

class decisionNode(
    val featureIndex: Int,
    val threshold: Double,
    val value:Int,
    val tnode: decisionNode,
    val fnode: decisionNode
) {
    def predict(x: Array[Double]): Int = {
        if(tnode != null && fnode != null) {
            if(x(featureIndex) >= threshold) tnode.predict(x)
            else fnode.predict(x)
        } else value
    }
    override def toString: String = {
        if(tnode != null && fnode != null) {
            s"col[$featureIndex]" + " >= " + threshold +
            s" ? ($tnode) : ($fnode)"
        } else s"class[$value]"
    }
}

class DecisionTreeII() {
    var tree: decisionNode = null
    var catColumns: Set[Int] = Set[Int]()
    var maxLayer: Int = 5

    private def log2(x: Double) = Math.log(x) / Math.log(2)

    private def uniqueCount(data: Array[(Int, Array[Double])]): Map[Int, Int] =
        data.groupBy(_._1).map(t => (t._1, t._2.size))

    private def entropy(data: Array[(Int, Array[Double])]): Double = {
        val dataSize = data.size.toDouble
        uniqueCount(data).map { case (k, v) =>
            val p = v / dataSize
            -p * log2(p)
        }.sum
    }

    private def buildtree(data: Array[(Int, Array[Double])], layer: Int = maxLayer): decisionNode = {
        val currentScore: Double = entropy(data)
        var bestGain: Double = 0
        var bestColumn: Int = 0
        var bestValue: Double = 0
        var bestTrueData = Array[(Int, Array[Double])]()
        var bestFalseData = Array[(Int, Array[Double])]()

        val dataSize = data.size.toDouble
        val columnSize: Int = data.head._2.size
        for (col <- 0 until columnSize) {
            var valueSet: Set[Double] = Set()
            for (d <- data) valueSet += d._2(col)
            for (value <- valueSet) {
                val (tData, fData) = data.partition { d =>
                    if(catColumns.contains(col)) d._2(col) == value
                    else d._2(col) >= value
                }
                val p = tData.size / dataSize
                val gain = currentScore - p * entropy(tData) - (1 - p) * entropy(fData)
                if (gain > bestGain && tData.size > 0 && fData.size > 0) {
                    bestGain = gain
                    bestColumn = col
                    bestValue = value
                    bestTrueData = tData
                    bestFalseData = fData
                }
            }
        }
        if (bestGain > 0 && layer > 0) {
            val tnode = buildtree(bestTrueData, layer - 1)
            val fnode = buildtree(bestFalseData, layer - 1)
            new decisionNode(bestColumn, bestValue, -1,tnode, fnode)
        } else new decisionNode(-1, -1,uniqueCount(data).maxBy(_._2)._1, null, null)
    }

     def train(data: Array[(Int, Array[Double])]): Boolean = try{
        tree = buildtree(data)
        true
    } catch { case e: Exception =>
        Console.err.println(e)
        false
    }

    def predict(x: Array[Array[Double]]): Array[Int] = x.map(xi => tree.predict(xi))
}

object DecisionTreeII{
    def main(args: Array[String]): Unit = {
        val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
          .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
          .toArray

        val data = dataS.map(x => (x.apply(4).toInt,x.slice(0,4)))
//        data.foreach(x => println(x._2.mkString("-")))
        val dtree = new DecisionTreeII
        dtree.train(data)
        println(dtree.tree.toString)

       val pred =  dtree.predict(data.map(_._2)).zip(data.map(_._1)).map(x => if(x._1 == x._2) 1 else 0 )
        println("准确率为: "+pred.sum.toDouble / data.size)

    }
}
