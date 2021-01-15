package com.strings.model.cluster

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{abs, exp}
import com.strings.data.Data
import com.strings.utils.Utils

import scala.collection.mutable.ArrayBuffer

/**
 * 幂迭代聚类算法
 */

class PIC(val k:Int = 3,
          val max_iter:Int = 100,
          val seed:Long = 1234L,
          val tolerance: Double = 1e-4,
          val sigma:Double = 100.0) {

  def normvec(x:DenseVector[Double]):DenseVector[Double] = {
    val norm1 = sum(x.map(math.abs(_)))
    x :/ norm1
  }

  def infinteNorm(x:DenseVector[Double]):Double = {
    max(abs(x))
  }

  def fit(X:List[DenseVector[Double]]):List[Int] = {
    val mat:DenseMatrix[Double] = DenseMatrix(Utils.pair_distance(X):_*)
    val n = X.length

    val W=  exp(mat :/ (-2*sigma*sigma))
    val n_samples = X.length
    for(i <- Range(0,n_samples)){
      W(i to i,i) := 0.0
    }

    var v0:DenseVector[Double] = DenseVector.zeros(n)
    var v1:DenseVector[Double] = normvec(DenseVector.rand(n))

    var d0 = v0
    var d1 = v1
    val threshold = 1e-5 / n
    while (infinteNorm(d1 - d0) >= threshold){
      v0 = v1
      v1 = normvec(W * v1)
      d0 = d1
      d1 = abs(v1 - v0)
    }


    val samples = ArrayBuffer[DenseVector[Double]]()
    for(i <- 0 until v1.length){
      val sample_i =  DenseVector(v1(i))
      samples.append(sample_i)
    }

    val kmeans = new Kmeans(k = k)
    kmeans.train(samples.toList)
    kmeans.label
  }

}

object PIC{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val pic = new PIC(max_iter = 100)
    val label =  pic.fit(data)
    println(label)

  }
}

