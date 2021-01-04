package com.strings.model.cluster

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmin, diag}
import scala.collection.mutable.ArrayBuffer
import breeze.stats.mean
import com.strings.data.Data
import com.strings.utils.MatrixUtils

/**
 *  Another version of Hierarchical Cluster implement.
 */

class Agnes(n_clusters:Int) {

  val clusters: ArrayBuffer[ArrayBuffer[Int]] = new ArrayBuffer[ArrayBuffer[Int]]()

  def fit(X:DenseMatrix[Double]):Unit = {
    val n_samples = X.rows
    for (i <- Range(0, n_samples)) {
      val res_i: ArrayBuffer[Int] = new ArrayBuffer[Int]()
      res_i.append(i)
      clusters.append(res_i)
    }

    for (j <- Range(n_samples, n_clusters, -1)) {
      val centers = clusters.map { t =>
        val aa = t.map(X.t(::, _))
        val mm = DenseMatrix(aa: _*)
        val me = mean(mm, Axis._0).t
        me
      }
      val distance:ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()
      for (j <- 0 until centers.length) {
        val d = centers.map(f => MatrixUtils.euclidean_distance(f, centers(j)))
         distance.append(DenseVector(d.toArray))
      }
      val diagMat = diag(DenseVector.fill(j){Double.MaxValue})

      val near_indexes = argmin(DenseMatrix(distance:_*) :+ diagMat)

      clusters(near_indexes._1).append(near_indexes._2)
      clusters.remove(near_indexes._2)

    }
  }

 def predict(X: DenseMatrix[Double]): DenseVector[Int] ={
   val y = Array.fill(X.rows)(0)
   for(i <- Range(0,clusters.length)){
     for(j <- clusters(i)){
       y(j) = i
     }
   }
   DenseVector(y)
 }

}

object Agnes {
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0, 4))
    val dd = DenseMatrix(data: _*)
    val ag = new Agnes(n_clusters = 3)
    ag.fit(dd)

    val label = ag.predict(dd)
    println(label)


  }
}
