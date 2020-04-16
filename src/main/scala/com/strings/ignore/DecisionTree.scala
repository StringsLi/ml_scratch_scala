
package com.strings.ignore

class DecisionNode(
    val col: Int,
    val v: Double,
    val tnode: DecisionNode,
    val fnode: DecisionNode,
    val r: Int = 0,
    val cats: Set[Int] = Set[Int]()
) {
    def predict(x: Array[Double]): Int = {
        if(tnode != null && fnode != null) {
            if((!cats.contains(col) && x(col) > v) || x(col) == v) tnode.predict(x)
            else fnode.predict(x)
        } else r
    }
    override def toString: String = {
        if(tnode != null && fnode != null) {
            s"col[$col]" + (if(cats.contains(col)) " == " else " >= ") + v +
            s" ? ($tnode) : ($fnode)"
        } else s"class[$r]"
    }
}

class DecisionTree() {
    var tree: DecisionNode = null
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

    private def buildtree(data: Array[(Int, Array[Double])], layer: Int = maxLayer): DecisionNode = {
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
            new DecisionNode(bestColumn, bestValue, tnode, fnode)
        } else new DecisionNode(0, 0, null, null, uniqueCount(data).maxBy(_._2)._1)
    }

     def train(data: Array[(Int, Array[Double])]): Boolean = try {
        tree = buildtree(data)
        true
    } catch { case e: Exception =>
        Console.err.println(e)
        false
    }




    def predict(x: Array[Array[Double]]): Array[Int] = x.map(xi => tree.predict(xi))
}

object DecisionTree{
    def main(args: Array[String]): Unit = {
        val dataS = scala.io.Source.fromFile("D:/data/iris.csv").getLines().toSeq.tail
          .map{_.split(",").filter(_.length() > 0).map(_.toDouble)}
          .toArray

        val data = dataS.map(x => (x.apply(4).toInt,x.slice(0,4)))

//        data.foreach(x => println(x._2.mkString("-")))

        val dtree = new DecisionTree
//        dtree.buildtree(data,4)

        dtree.train(data)

        println(dtree.tree.toString)

    }
}
