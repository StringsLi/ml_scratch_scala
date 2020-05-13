package com.strings.model.cluster

import breeze.linalg.{DenseMatrix, DenseVector, squaredDistance}
import com.strings.data.Data
import com.strings.model.metric.Metric

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import util.control.Breaks.breakable
import util.control.Breaks.break

/**
 *  Partitioning (clustering) of the data into k clusters “around medoids”, a more robust version of K-means.
 *
 * @param k nums of cluster
 */


class PAM(k:Int = 2,seed:Long = 1234L) {

  def _init_random_medoids(X: DenseMatrix[Double]): IndexedSeq[DenseVector[Double]] ={
    val n_samples = X.rows
    val data = (0 until n_samples).map(X.t(::,_))
    val rng  = new Random(seed)
    rng.shuffle(data).take(k)
  }

  def _closet_medoid(sample:DenseVector[Double],medoids:IndexedSeq[DenseVector[Double]]): Int ={
    val distWithIndex =  medoids.zipWithIndex.map(x =>
      (squaredDistance(x._1,sample),x._2)
    ).minBy(_._1)
    distWithIndex._2
  }

  def _create_clusters(X:DenseMatrix[Double],medoids:IndexedSeq[DenseVector[Double]]): Array[Array[Int]] ={
    val clusterss = new Array[Int](X.rows)
    val data = (0 until X.rows).map(X.t(::,_))
    for((sample,inx) <- data.zipWithIndex){
      val medoid_i = _closet_medoid(sample,medoids)
      clusterss(inx) = medoid_i
    }
    clusterss.zipWithIndex.groupBy(_._1).toArray.sortBy(_._1).map(_._2.map(_._2))
  }

  def _calculate_cost(X:DenseMatrix[Double],clusters:Array[Array[Int]],medoids:IndexedSeq[DenseVector[Double]]):Double={
    var cost = 0.0
    val data = (0 until X.rows).map(X.t(::,_))
    for((cluster,i) <- clusters.zipWithIndex){
      val medoid = medoids(i)
      for(sample_i <- cluster){
        cost += squaredDistance(data(sample_i),medoid)
      }
    }
    cost
  }

  def _get_cluster_labels(clusters:Array[Array[Int]],X:DenseMatrix[Double]): Array[Int] ={
    val y_pred = Array.fill(X.rows)(0)
    for(cluster_i <- 0 until clusters.length){
      val cluster = clusters(cluster_i)
      for(sample_i <- cluster){
        y_pred(sample_i) = cluster_i
      }
    }
    y_pred
  }
  def _get_no_medoids(X:DenseMatrix[Double],medoids:IndexedSeq[DenseVector[Double]]): Array[DenseVector[Double]] ={
    val non_medoids:ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()
    val data = (0 until X.rows).map(X.t(::,_))

    for(sample <- data){
      if(!medoids.contains(sample)) non_medoids.append(sample)
    }
    non_medoids.toArray
  }

  def predict(X:DenseMatrix[Double]): Array[Int] ={
    var medoids = _init_random_medoids(X)
    val clusters = _create_clusters(X,medoids)
    var cost = _calculate_cost(X, clusters, medoids)
    breakable {
      while (true) {
        var best_medoids = medoids
        var lowest_cost = cost
        for (medoid <- medoids) {
          val non_medoids = _get_no_medoids(X, medoids)
          for (sample <- non_medoids) {
            val new_medoids = new Array[DenseVector[Double]](medoids.length)
            for (i <- 0 until medoids.length) {
              new_medoids(i) = medoids(i)
            }
            val inx: IndexedSeq[Int] = medoids.indices.filter(i => medoids(i) == medoid)
            inx.foreach(i => new_medoids(i) = sample)

            val new_clusters = _create_clusters(X, new_medoids)
            val new_cost = _calculate_cost(X, new_clusters, new_medoids)

            if (new_cost < lowest_cost) {
              lowest_cost = new_cost
              best_medoids = new_medoids
            }
          }
        }
        if (lowest_cost < cost) {
          cost = lowest_cost
          medoids = best_medoids
        } else {
          break()
        }
      }
    }
    val finaly_clusters = _create_clusters(X,medoids)
    _get_cluster_labels(finaly_clusters,X)
  }

}

object PAM{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).toList
    val target = irisData.map(_.apply(4))

    val pam = new PAM(k = 3)

    val pred = pam.predict(DenseMatrix(data:_*))
    println(pred.toList)
    val acc =  Metric.accuracy(pred.map(_.toDouble),target) * 100
    println(f"准确率为: $acc%-5.2f%%")
  }
}
