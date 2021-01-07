package com.strings.model.cluster

import breeze.linalg.{Axis, DenseMatrix, DenseVector, diag, eig, inv, sum}
import breeze.numerics.{exp, log}
import com.strings.data.Data
import com.strings.utils.Utils

/**
 * 谱聚类
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
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _))
    val eigenValues = eigen.eigenvalues.toArray
    val topEigenvectors = eigenvectors.zip(eigenValues).sortBy(x => -scala.math.abs(x._2)).map(_._1).take(k)

    val kmeans = new Kmeans(k = k)
    kmeans.train(topEigenvectors.toList)
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
