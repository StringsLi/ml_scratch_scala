package com.strings.model.cluster

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import com.strings.data.Data
import breeze.stats.variance
import scala.util.control.Breaks
import scala.collection.mutable.ArrayBuffer

class BisectingKMeans(val k: Int = 3,
                      val max_iter: Int = 100,
                      val seed: Long = 1234L,
                      val tolerance: Double = 1e-3) {


  val clusters = ArrayBuffer[Array[Int]]()

  def fit(X: List[DenseVector[Double]]): Unit = {
    var data = X
    Breaks.breakable {
      while (true) {
        val model = new Kmeans(2, max_iter, seed, tolerance)
        model.train(data)
        val label_k = model.label

        clusters.append(label_k.zipWithIndex.filter(x => x._1 == 0).map(_._2).toArray)
        clusters.append(label_k.zipWithIndex.filter(x => x._1 == 1).map(_._2).toArray)
        if (clusters.size == k) {
          Breaks.break()
        }
        val sse = for (cluster <- clusters) yield {
          val dd = data.zipWithIndex.filter(x => cluster.contains(x._2)).map(_._1)
          variance(DenseMatrix(dd: _*))
        }
        val sseVector = DenseVector(sse: _*)
        val index = argmax(sseVector) // 方差最较大的簇编号
        data = data.zipWithIndex.filter(x => clusters(index).contains(x._2)).map(_._1)
        clusters.remove(index)
      }
    }
  }
  def predict(X:List[DenseVector[Double]]):Array[Int] = {
   val  label = new Array[Int](X.length)
    for (i <- clusters.indices) {
      for (j <- clusters(i)) {
        label(j) = i
      }
    }
    label
  }


}


object BisectingKMeans {
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0, 4)).map(DenseVector(_)).toList
    val kmeans = new BisectingKMeans(max_iter = 100)
    kmeans.fit(data)
    val label = kmeans.predict(data)
    println(label.toList)

  }
}
