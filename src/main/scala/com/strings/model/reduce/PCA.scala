package com.strings.model.reduce


import breeze.linalg.{DenseMatrix, eig}
import com.strings.data.Data
import com.strings.utils.{FileUtils, MatrixUtils}

object PCA {

  def transform(X: DenseMatrix[Double], numComponents: Int):DenseMatrix[Double] = {
    val scaterMatrix = MatrixUtils.calculate_covariance_matrix(X)
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
    val transData = transform(denseMatrix,2)

    val file = "D:\\data\\iris_pca2.txt"
    FileUtils.writeFile(transData,file)
  }

}
