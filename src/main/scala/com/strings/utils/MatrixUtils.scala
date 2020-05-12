package com.strings.utils

import breeze.linalg.{*, DenseMatrix}
import breeze.stats.mean

object MatrixUtils extends  Serializable {

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



}
