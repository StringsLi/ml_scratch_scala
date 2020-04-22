package com.strings.model.reduce

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, Transpose, diag, eig, eigSym, squaredDistance}
import breeze.numerics.sqrt
import breeze.stats.mean
import com.strings.data.Data

/**
 * Multiple dimensional scaling for dimemsion reducing
 */

object MDS {

  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x => DenseVector(x.slice(0,4))).toList
    val transformData = mds(data,3)
    println(transformData)
  }

  def pair_distance(X:List[DenseVector[Double]]) = {
    val nSample = X.length
    val pairDist:Array[Array[Double]] = Array.ofDim[Double](nSample,nSample)

    for(i <- 0 until nSample){
      for(j <- 0 until nSample){
        pairDist(i)(j) = squaredDistance(X(i),X(j))
      }
    }
    DenseMatrix(pairDist:_*)
  }

  def mds(data:List[DenseVector[Double]],k:Int) = {
    val dist:DenseMatrix[Double] = pair_distance(data)
    val M_r:Transpose[DenseVector[Double]] = mean(dist(::,*))
//    val M_r1:Transpose[DenseVector[Double]]  = mean(dist, Axis._0)
    val M_c:DenseVector[Double] = mean(dist(*,::))
    val meanDist:Double = mean(dist)

    val B = (dist(::,*).map(x => x - M_c - M_r.t) :+ meanDist) :* (-0.5)

    val eigen = eig(B)
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _))
    val eigenValues = eigen.eigenvalues.toArray
    val topEigenvectorNvalues = eigenvectors.zip(eigenValues).sortBy(x => -scala.math.abs(x._2)).take(k)

    val gamma = sqrt(diag(DenseVector(topEigenvectorNvalues.map(_._2):_*)))
    val topEigMat = DenseMatrix(topEigenvectorNvalues.map(_._1):_*)

    topEigMat.t * gamma
  }

}
