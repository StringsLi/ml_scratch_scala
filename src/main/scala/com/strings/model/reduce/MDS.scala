package com.strings.model.reduce

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, Transpose, diag, eig, eigSym, squaredDistance}
import breeze.numerics.sqrt
import breeze.stats.mean
import com.strings.data.Data
import com.strings.utils.{FileUtils, Utils}

/**
 * Multiple dimensional scaling for dimemsion reducing
 */

object MDS {

  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x => DenseVector(x.slice(0,4))).toList
    val transformData = mds(data,2)
    println(transformData)
    val file = "D:\\data\\iris_mds.txt"
    FileUtils.writeFile(transformData,file)
  }


  def mds(data:List[DenseVector[Double]],k:Int):DenseMatrix[Double] = {
    val dist:DenseMatrix[Double] = DenseMatrix(Utils.pair_distance(data):_*)
    val M_r:Transpose[DenseVector[Double]] = mean(dist(::,*))  //mean(dist, Axis._0)
    val M_c:DenseVector[Double] = mean(dist(*,::))
    val meanDist:Double = mean(dist)

    val B1:DenseMatrix[Double] = dist(::,*).map(x => x - M_c)
    val B = (B1(*,::).map(x => x - M_r.t )  :+ meanDist):* (-0.5)
//    val B = (dist(::,*).map(x => x - M_c - M_r.t) :+ meanDist) :* (-0.5)  //计算出来不是矩阵B的每一列以及每一列求和均为0

    val eigen = eig(B)
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _))
    val eigenValues = eigen.eigenvalues.toArray
    val topEigenvectorNvalues = eigenvectors.zip(eigenValues).sortBy(x => -scala.math.abs(x._2)).take(k)

    val gamma = sqrt(diag(DenseVector(topEigenvectorNvalues.map(_._2):_*)))
    val topEigMat = DenseMatrix(topEigenvectorNvalues.map(_._1):_*)

    topEigMat.t * gamma
  }

}
