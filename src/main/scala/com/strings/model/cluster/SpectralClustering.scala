package com.strings.model.cluster

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, diag, eig, inv, sum}
import breeze.numerics.{exp, log}
import com.strings.data.Data
import com.strings.utils.Utils

import scala.collection.mutable.ArrayBuffer

/**
 * 谱聚类
 *
 * Compute the rbf (gaussian) kernel between X and Y::
 * K(x, y) = exp(-sigma ||x-y||^2)
 *
 */

class SpectralClustering(val k:Int = 3,
                         val max_iter:Int = 100,
                         val seed:Long = 1234L,
                         val tolerance: Double = 1e-4,
                         val sigma:Double = 100.0) {

  def fit(X:List[DenseVector[Double]]) = {
    val mat:DenseMatrix[Double] = DenseMatrix(Utils.pair_distance(X):_*)
    val W=  exp(mat :/ (-2*sigma*sigma))
    val n_samples = X.length
    for(i <- Range(0,n_samples)){
      W(i to i,i) := 0.0
    }
    val D = diag(sum(W,Axis._1))
    val L = inv(D) * (D - W)
    val eigen = eig(L)

    val eigenvectors = (0 until k).map(eigen.eigenvectors(::, _))
    val U = DenseMatrix(eigenvectors:_*)

    val samples = ArrayBuffer[DenseVector[Double]]()
    for(i <- 0 until U.cols){
      val sample_i =  U(::,i)
      samples.append(sample_i)
    }
    val kmeans = new Kmeans(k = k)
    kmeans.train(samples.toList)
     kmeans.label
  }

}

object SpectralClustering{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val spc = new SpectralClustering(max_iter = 100)
   val label =  spc.fit(data)
    println(label)

  }
}
