package com.strings.model.reduce

import breeze.linalg.{DenseMatrix, DenseVector, diag, eig, pinv, sum}
import com.strings.data.Data
import com.strings.utils.{FileUtils, Utils}

/**
 * scala 版的lle好像有些问题，需要再调试
 * @param k  the number of neighbors
 */

class LLE(var k:Int = 50) {

  /**
   * Claculates k nearest neighbors of each line in given matrix
   * @param data Matrix N*p where each row is datapoint
   * @return nearest - neigbor index matrix N*k
   */
  def knn(data:List[DenseVector[Double]]):Array[Array[Int]] = {
    val dist = Utils.pair_distance(data)
    dist.map(x => x.zipWithIndex.sortBy(_._1).map(_._2).slice(1,k+1))
  }

  def findW(X:List[DenseVector[Double]],near:Array[Array[Int]]):Array[Array[Double]] = {
    val n = X.length
    val W = Array.fill(n,n)(0.0)
    for(i <- Range(0,n)){
      val idx:Array[Int] = near(i)
      val neigbor = idx.map(X(_))
      val x_i:DenseVector[Double] = X(i)
      val diff = DenseMatrix(neigbor.map(f => f - x_i):_*)
      val gram = diff *  diff.t
      val w = pinv(gram) * DenseVector.fill(k)(1.0)
      val ww = w :/ sum(w)
      var count = 0
      for(j <- idx){
        W(i)(j) = ww(count)
        count += 1
      }
    }
    W
  }

  /**
   *
   * @param X NxD data matrix.
   * @param d dimension
   *          Nxd reduced data matrix.
   */
  def transform(X:List[DenseVector[Double]],d:Int):DenseMatrix[Double] = {
    val near = knn(X)
    val W = findW(X,near)
    val n = W.length
    val eye:DenseVector[Double] = DenseVector.fill(n)(1.0)
    val diffMatrix:DenseMatrix[Double] = diag(eye) - DenseMatrix(W:_*)
    val M = diffMatrix.t * diffMatrix
    val eigen = eig(M)
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _).toDenseMatrix.t)
    val topEigenvectorsNvalue = eigenvectors.zip(eigen.eigenvalues.toArray).sortBy(x => scala.math.abs(x._2)).slice(1,d+1)
    DenseMatrix.horzcat(topEigenvectorsNvalue.map(_._1):_*)
  }
}

object LLE{
  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x => DenseVector(x.slice(0,4))).toList
    val lle = new LLE()
    val W = lle.transform(data,2)
    val file = "D:\\data\\iris_lle.txt"
    FileUtils.writeFile(W,file)
  }
}