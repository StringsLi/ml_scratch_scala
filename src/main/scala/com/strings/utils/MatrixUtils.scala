package com.strings.utils

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.abs
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

  def euclidean_distance(x1: DenseVector[Double], x2: DenseVector[Double]): Double = {
    require(x1.length == x2.length)
    math.sqrt(sum((x1 :- x2) :* (x1 :- x2)))
  }

  def manhattan_distance(x1: DenseVector[Double], x2: DenseVector[Double]): Double = {
    require(x1.length == x2.length)
    sum(abs(x1 :- x2))
  }


}
