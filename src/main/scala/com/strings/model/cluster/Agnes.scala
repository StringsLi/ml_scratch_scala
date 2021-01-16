package com.strings.model.cluster

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmin, diag}
import scala.collection.mutable.ArrayBuffer
import breeze.stats.mean
import com.strings.data.Data
import com.strings.utils.MatrixUtils

/**
 *  Another version of Hierarchical Cluster implement.
 */

class Agnes(k:Int) {

  val clusters: ArrayBuffer[ArrayBuffer[Int]] = new ArrayBuffer[ArrayBuffer[Int]]()

  def fit(X:DenseMatrix[Double]):Unit = {
    val n_samples = X.rows
    for (i <- Range(0, n_samples)) {
      val res_i: ArrayBuffer[Int] = new ArrayBuffer[Int]()
      res_i.append(i)
      clusters.append(res_i)
    }

    for (j <- Range(n_samples, k, -1)) {
      val centers = clusters.map { r =>
        mean(DenseMatrix(r.map(X.t(::, _)): _*), Axis._0).t
      }
      val distance:ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()
      for (j <- centers.indices) {
        val d = centers.map(f => MatrixUtils.euclidean_distance(f, centers(j)))
         distance.append(DenseVector(d.toArray))
      }
      val diagMat = diag(DenseVector.fill(j){Double.MaxValue})
      val nearIndexs = argmin(DenseMatrix(distance:_*) :+ diagMat)

      clusters(nearIndexs._1).append(clusters(nearIndexs._2):_*)
      clusters.remove(nearIndexs._2)

    }
  }

 def predict(X: DenseMatrix[Double]): DenseVector[Int] ={
   val y = Array.fill(X.rows)(0)
   for(i <- clusters.indices){
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
    val mat = DenseMatrix(data: _*)
    val agnes = new Agnes(k = 3)
    agnes.fit(mat)

    val label = agnes.predict(mat)
    println(label)


  }
}
