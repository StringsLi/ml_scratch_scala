package com.strings.model.reduce

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, Transpose, diag, eig}
import breeze.numerics.sqrt
import breeze.stats.mean
import com.strings.data.Data
import com.strings.utils.{FileUtils, Utils}

object Isomap {

  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x => DenseVector(x.slice(0,4))).toList
    val W = isomap(data,2,15)
    val file = "D:\\data\\iris_isomap.txt"
    FileUtils.writeFile(W,file)
  }

  def floyd(data:Array[Array[Double]],k:Int):Array[Array[Double]] = {
    val n = data.length
    val D1 = Array.fill(n,n)(Double.PositiveInfinity)
    val m_arg:Array[Array[Int]] = data.map(x => x.zipWithIndex.sortBy(-_._1).map(_._2))

    m_arg.zipWithIndex.foreach{
      case(arg,inx) =>
        for(j <- arg.take(k)){
          D1(inx)(j) = data(inx)(j)
      }
    }

    for(i <- Range(0,n)){
      for(j <- Range(0,n)) {
        for (k <- Range(0, n)) {
          if(D1(i)(k) + D1(k)(j) < D1(i)(j)){
            D1(i)(j) = D1(i)(k) + D1(k)(j)
          }
        }
      }
    }
    D1
  }

  /**
   *
   * @param data 成对距离矩阵
   * @param start jjf
   */
  def Dijkstra(data:Array[Array[Double]],start:Int):Unit = {
    val n = data.length
    val col:Array[Double] = Array.ofDim(n)
    for(i <- Range(0,n)){
      col(i) = data(start)(i)
    }
    var rem = n - 1
    while(rem > 0){
      val i:Int = col.zipWithIndex.maxBy(_._1)._2
      val temp = data(start)(i)
      for(j <- Range(0,n)){
        if(data(start)(j) > temp + data(i)(j)){
           data(start)(j) = temp + data(i)(j)
           data(j)(start) = data(start)(j)
        }
      }
      rem = rem - 1
      col(i) = Double.MaxValue
    }
  }

  def mds(dist:DenseMatrix[Double],k:Int):DenseMatrix[Double] = {
    val M_r:Transpose[DenseVector[Double]]  = mean(dist, Axis._0)
    val M_c:DenseVector[Double] = mean(dist,Axis._1)
    val meanDist:Double = mean(dist)
    val B1:DenseMatrix[Double] = dist(::,*).map(x => x - M_c)
    val B = (B1(*,::).map(x => x - M_r.t )  :+ meanDist):* (-0.5)
//    val B = (dist(::,*).map(x => x - M_c - M_r.t) :+ meanDist) :* (-0.5)
    val eigen = eig(B)
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _))
    val eigenValues = eigen.eigenvalues.toArray
    val topEigenvectorNvalues = eigenvectors.zip(eigenValues).sortBy(x => -scala.math.abs(x._2)).take(k)

    val gamma = sqrt(diag(DenseVector(topEigenvectorNvalues.map(_._2):_*)))
    val topEigMat = DenseMatrix(topEigenvectorNvalues.map(_._1):_*)

    topEigMat.t * gamma
  }


  def isomap(data:List[DenseVector[Double]],numComponents:Int,k:Int = 10): DenseMatrix[Double] ={
    val n = data.length
    val mat_dist = Utils.pair_distance(data)
    val knear = Array.fill(n,n)(1.0)
    for(idx <- Range(0,n)){
      val topk:Array[Int] = mat_dist.apply(idx).zipWithIndex.sortBy(_._1).map(_._2).take(k)
      for(j <- topk){
        knear(idx)(j) = mat_dist(idx)(j)
      }
    }
    for(i <- Range(0,n)){
      Dijkstra(knear,i)
    }
    mds(DenseMatrix(knear:_*),numComponents)
  }

  def isomap2(data:List[DenseVector[Double]],numComponents:Int,k:Int = 10): DenseMatrix[Double] ={
    val mat_dist = Utils.pair_distance(data)
    val D_floyd = floyd(mat_dist,k)
    mds(DenseMatrix(D_floyd:_*),numComponents)
  }

}
