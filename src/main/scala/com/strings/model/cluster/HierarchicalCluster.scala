package com.strings.model.cluster

import breeze.linalg.{DenseMatrix, DenseVector}
import com.strings.data.Data
import com.strings.utils.MatrixUtils

import scala.collection.mutable.ArrayBuffer


case class ClusterNode(vector:DenseVector[Double],
                       id:Int,
                       left:ClusterNode = null,
                       right:ClusterNode = null,
                       distance:Double = -1.0,
                       count:Int = 1)

class HierarchicalCluster(k:Int) {

  var labels:DenseVector[Int] = _

  def fit(X:DenseMatrix[Double]): Unit ={
    val n_samples = X.rows
    val n_features = X.cols
    val X_arr = (0 until n_samples).map(X.t(::,_))
    val nodes:ArrayBuffer[ClusterNode] = new ArrayBuffer[ClusterNode]()
    for((sample,inx) <- X_arr.zipWithIndex){
      nodes.append(ClusterNode(sample,inx))
    }
    labels = DenseVector.ones[Int](n_samples) :* (-1)
    var distances: Map[(Int, Int), Double] = Map()
    var current_cluster_id = -1
    while (nodes.length > k){
      var min_dist = Double.MaxValue
      val nodes_len = nodes.length
      var closest_part:(Int, Int) = 0 -> 0
      for(i <- 0 until nodes_len - 1){
        for(j <- i+1 until nodes_len){
          val d_key = nodes(i).id -> nodes(j).id
          if(!distances.contains(d_key)){
            distances += (d_key -> MatrixUtils.euclidean_distance(nodes(i).vector,nodes(j).vector))
          }
          val d = distances(d_key)
          if(d < min_dist){
            min_dist = d
            closest_part = i -> j
          }
        }
      }

      val part1 = closest_part._1
      val part2 = closest_part._2
      val node1 = nodes(part1)
      val node2 = nodes(part2)
      val new_vec = DenseVector.ones[Double](n_features)
      for(i <- 0 until n_features){
        new_vec(i) = (node1.vector(i) * node1.count + node2.vector(i) * node2.count)/
          (node1.count + node2.count)
      }
      val new_count = node1.count + node2.count
      val new_node = ClusterNode(new_vec,current_cluster_id,node1,node2, min_dist,new_count)

      current_cluster_id -= 1
      nodes.remove(part2)
      nodes.remove(part1)
      nodes.append(new_node)
    }
    calc_label(nodes)

  }

  def calc_label(nodes:ArrayBuffer[ClusterNode]): Unit ={
    for((node,inx) <- nodes.zipWithIndex){
      leaf_traveral(node,inx)
    }
  }

  def leaf_traveral(node:ClusterNode,label:Int): Unit ={
    if(node.left == null && node.right == null){
      labels(node.id) = label
    }
    if(node.left != null){
      leaf_traveral(node.left,label)
    }
    if(node.right != null){
      leaf_traveral(node.right,label)
    }
  }
}

object HierarchicalCluster{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4))
    val dd = DenseMatrix(data:_*)
    val hc = new HierarchicalCluster(k=3)
    hc.fit(dd)
    println(hc.labels)
  }
}
