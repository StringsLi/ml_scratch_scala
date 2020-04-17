package com.strings.model.tree

import scala.collection.mutable

abstract class DecisionTree(val min_samples_split:Int=2,
                            val min_impurity:Double=1e-7,
                            val max_depth:Int = 5) {
  var root: DecisionNode = null
  var catColumns: Set[Int] = Set[Int]()
  var _impurity_calculation: CalcInfoGain = _
  var _leaf_value_calc: LeafValueCalc = _

  def init_impurity_calc(): Unit

  def init_leaf_value_calc(): Unit

  def fit(data: Array[(Double, Array[Double])]): Unit = {
    init_impurity_calc()
    init_leaf_value_calc()
    root = buildtree(data)
  }

  def fit(X: Array[Array[Double]], y: Array[Double]):Unit={
    val data = y.zip(X)
    fit(data)
  }

  private def buildtree(data: Array[(Double, Array[Double])], current_depth: Int = 0): DecisionNode = {
    var bestGain: Double = 0
    var bestColumn: Int = 0
    var bestValue: Double = 0
    var bestTrueData = Array[(Double, Array[Double])]()
    var bestFalseData = Array[(Double, Array[Double])]()
    val columnSize: Int = data.head._2.size
    val nSamples: Int = data.size
    if (nSamples >= min_samples_split && current_depth <= max_depth) {
      for (col <- 0 until columnSize) {
        var valueSet: Set[Double] = Set()
        for (d <- data) valueSet += d._2(col)
        for (value <- valueSet) {
          val (tData, fData) = data.partition { d =>
            if (catColumns.contains(col)) d._2(col) == value
            else d._2(col) >= value
          }
          val gain = _impurity_calculation.impurity_calculation(data.map(_._1), tData.map(_._1), fData.map(_._1))
          if (gain > bestGain && tData.size > 0 && fData.size > 0) {
            bestGain = gain
            bestColumn = col
            bestValue = value
            bestTrueData = tData
            bestFalseData = fData
          }
        }
      }
    }
    if (bestGain > min_impurity) {
      val tnode: DecisionNode = buildtree(bestTrueData, current_depth + 1)
      val fnode: DecisionNode = buildtree(bestFalseData, current_depth + 1)
      new DecisionNode(bestColumn, bestValue, -1, tnode, fnode)
    } else {
      val leafValue = _leaf_value_calc.leafCalc(data.map(_._1))
      new DecisionNode(-1, -1, leafValue, null, null)
    }
  }

  case class dotNode(parentId: Int, id: Int, node: DecisionNode)

  def dot(root: DecisionNode): String = {
    val builder = new StringBuilder
    builder.append("digraph DecisionTree {\n node [shape=box, style=\"filled, rounded\", color=\"black\", fontname=helvetica];\n edge [fontname=helvetica];\n")
    var n = 0 // number of nodes processed
    val queue: mutable.Queue[dotNode] = new scala.collection.mutable.Queue[dotNode]
    queue.enqueue(dotNode(-1, 0, root))
    while (!queue.isEmpty) { // Dequeue a vertex from queue and print it
      val dnode = queue.dequeue()
      val id = dnode.id
      val parent = dnode.parentId
      val node: DecisionNode = dnode.node
      // leaf node
      if (node.fnode == null && node.tnode == null) {
        builder.append(" %d [label=<class = %s>, fillcolor=\"#00000000\", shape=ellipse];\n".format(id, node.value))
      }
      else {
        builder.append(" %d [label=<featur: %d &le; %s<br/>>, fillcolor=\"#00000000\"];\n".format(id, node.featureIndex, node.threshold))
      }
      // add edge
      if (parent >= 0) {
        builder.append(' ').append(parent).append(" -> ").append(id)
        // only draw edge label at top
        if (parent == 0) if (id == 1) builder.append(" [labeldistance=2.5, labelangle=45, headlabel=\"True\"]")
        else builder.append(" [labeldistance=2.5, labelangle=-45, headlabel=\"False\"]")
        builder.append(";\n")
      }

      if (node.fnode != null) {
        n += 1
        queue.enqueue(dotNode(id, n, node.fnode))
      }

      if (node.tnode != null) {
        n += 1
        queue.enqueue(dotNode(id, n+1, node.tnode))
      }

    }
    builder.append("}")
    builder.toString
  }

  def predict(x: Array[Array[Double]]): Array[Double] = x.map(xi => root.predict(xi))
  def predict(x: Array[Double]): Double = root.predict(x)

}
