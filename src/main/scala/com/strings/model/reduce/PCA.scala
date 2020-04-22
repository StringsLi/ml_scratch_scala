package com.strings.model.reduce

import breeze.linalg.{*, DenseMatrix, eig}
import breeze.stats.mean
import com.strings.data.Data

object PCA {

  def calculate_covariance_matrix(X:DenseMatrix[Double]):DenseMatrix[Double] = {
    val meanVec = mean(X(::,*))
    val dim = X.cols
    val total_sample = X.rows
    var scatter_mat = DenseMatrix.zeros[Double](dim, dim)
    for (i <- 0 until total_sample) {
      scatter_mat += (X(i,::) - meanVec).t*(X(i,::) - meanVec)
    }
    val res = scatter_mat / (total_sample.toDouble - 1.0)
    res
  }

  def transform(X: DenseMatrix[Double], numComponents: Int):DenseMatrix[Double] = {
    val scaterMatrix = calculate_covariance_matrix(X)
    val eigen = eig(scaterMatrix)
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _))
    val eigenValues = eigen.eigenvalues.toArray
    val topEigenvectors = eigenvectors.zip(eigenValues).sortBy(x => -scala.math.abs(x._2)).map(_._1).take(numComponents)
    val W = DenseMatrix(topEigenvectors:_*)
    X * W.t
  }

  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x =>x.slice(0,4))
    val denseMatrix = DenseMatrix(data:_*)
    val transData = transform(denseMatrix,3)
    println(transData)
  }

}
