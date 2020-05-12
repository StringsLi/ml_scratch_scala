package com.strings.model.cluster

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import com.strings.data.Data
import com.strings.model.metric.Metric

import util.control.Breaks.breakable
import util.control.Breaks.break
import scala.collection.mutable.ArrayBuffer

class DBSCAN(eps:Double = 1.0,
             min_samples:Int = 5) {

  var visited_samples:ArrayBuffer[Int] = new ArrayBuffer[Int]()
  var neighbors: Map[Int, Array[Int]] = Map()
  var X:DenseMatrix[Double] = _
  var clusters:ArrayBuffer[Array[Int]] = new ArrayBuffer[Array[Int]]()

  def euclidean_distance(x1:DenseVector[Double],x2:DenseVector[Double]): Double ={
    math.sqrt(sum((x1 :- x2) :* (x1 :- x2)))
  }

  def _get_neighbors(sample_i:Int): Array[Int] ={
    val neighbors:ArrayBuffer[Int] = new ArrayBuffer[Int]()
    val X_arr = (0 until X.rows).map(X.t(::,_))
    val X_arr2 =  X_arr.indices.filter(i => i != sample_i).map(X_arr(_))
    for((_sample,inx) <- X_arr2.zipWithIndex){
      val dist = euclidean_distance(_sample,X_arr(sample_i))
      if(dist < eps){
        neighbors.append(inx)
      }
    }
    neighbors.toArray
  }

  def _expand_cluster(sample_i:Int, neighbor:Array[Int]): Array[Int] ={
    val cluster:ArrayBuffer[Int] = new ArrayBuffer[Int]()
    cluster.append(sample_i)
    for(neighbor_i <- neighbor){
      if(!visited_samples.contains(neighbor_i)){
        visited_samples.append(neighbor_i)
        neighbors += (neighbor_i -> _get_neighbors(neighbor_i) )
        if (neighbors(neighbor_i).size >= min_samples){   //        neighbors.get(neighbor_i).size 结果是1
          val expanded_cluster = _expand_cluster(neighbor_i,neighbors(neighbor_i))
          cluster.append(expanded_cluster:_*)
        }else{
          cluster.append(neighbor_i)
        }
      }
    }
    cluster.toArray
  }

  def _get_cluster_labels(): Array[Int] ={
    val labels = Array.fill(X.rows)(clusters.length)
    for((cluster,cluster_i) <- clusters.zipWithIndex){
      for(sample_i <- cluster){
        labels(sample_i) = cluster_i
      }
    }
    labels
  }

  def init(XX:DenseMatrix[Double]) {
    X = XX
  }

  def predict(XX:DenseMatrix[Double]): Array[Int] ={
    X = XX
    visited_samples = new ArrayBuffer[Int]()
    neighbors = Map()
    val n_samples = X.rows
    for(sample_i <- 0 until n_samples){
      breakable {
        if (visited_samples.contains(sample_i)) {
          break()
        }else{
          neighbors += (sample_i -> _get_neighbors(sample_i))
          if(neighbors.get(sample_i).size >= min_samples){
            visited_samples.append(sample_i)
          }
          val new_cluster = _expand_cluster(sample_i,neighbors(sample_i))
          clusters.append(new_cluster)
        }
      }
    }
    _get_cluster_labels()
  }

}

object DBSCAN{
  def main(args: Array[String]): Unit = {

    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).toList
    val target = irisData.map(_.apply(4))

    val dbscan = new DBSCAN(eps = .7,min_samples = 5)

    val pred = dbscan.predict(DenseMatrix(data:_*))
    println(pred.toList)
    val acc =  Metric.accuracy(pred.map(_.toDouble),target) * 100
    println(f"准确率为: $acc%-5.2f%%")
  }
}
