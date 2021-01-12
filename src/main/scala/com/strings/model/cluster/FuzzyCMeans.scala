package com.strings.model.cluster

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, argmax, max, sum}
import breeze.numerics.{abs, pow, sqrt}
import breeze.stats.distributions.Rand
import com.strings.data.Data
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks

class FuzzyCMeans(val k: Int = 4,
                    val fuzziness: Int = 2,
                    val epsilon: Double = 1e-6,
                    val seed: Long = 1234L,
                    val max_iter: Int = 10000) {

  private var U: DenseMatrix[Double] = _
  private var centroids:List[DenseVector[Double]] = _

  def initializeU(data: List[DenseVector[Double]]):DenseMatrix[Double] = {
    var U:DenseMatrix[Double] = DenseMatrix.rand(data.length, k,rand = Rand.gaussian)
    U = U(::, *) / sum(U, Axis._1)
    U
  }

  def distance(X: List[DenseVector[Double]], centroid:DenseVector[Double]):DenseVector[Double] ={
    val tmp:List[DenseVector[Double]] = X.map(x => pow(x - centroid,2))
      val mat = DenseMatrix(tmp:_*)
    sqrt(sum(mat,Axis._1))
  }


  def computeU(X: List[DenseVector[Double]]):DenseMatrix[Double] = {

    val n_samples = X.length
    var U:DenseMatrix[Double] = DenseMatrix.zeros(n_samples,k)

    for(i <- 0 until k){
      for(j <- 0 until k){
        val numerator = distance(X,centroids(i))
        val denominator = distance(X,centroids(j))
        U(::,i) :+= pow(numerator :/ denominator,2.0 /(fuzziness - 1))
      }
    }
    U = 1.0 / U
    U
  }


  def fit(data: List[DenseVector[Double]]):DenseVector[Int] = {
    val n_samples = data.length
    U = initializeU(data)
    var U_old = DenseMatrix.zeros[Double](n_samples,k)

    var iter_count = 1

    Breaks.breakable {
      for (_ <- Range(0, max_iter)) {
        iter_count += 1
        val centriods_new:ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()
        for(i <- 0 until k){
          val U_i = (0 until k).map(U(::,_)).apply(i)
          val U_i_Powerd = pow(U_i,fuzziness)
          val centroid = DenseMatrix(data:_*).t * U_i_Powerd
          val sum1:Double = sum(U_i_Powerd)
          centriods_new.append(centroid / sum1)
        }

        U_old = U.copy
        centroids = centriods_new.toList
        U = computeU(data)

        if (max(abs(U - U_old)) < epsilon) {
          Breaks.break()
        }
      }
    }

    argmax(U, Axis._1)
  }


}

object FuzzyCMeans {
  def main(args: Array[String]): Unit = {

    val irisData = Data.irisData
    val data = irisData.map(_.slice(0, 4)).map(DenseVector(_)).toList
    val kmeans = new FuzzyCMeans(k = 3,max_iter = 5000)
    val label = kmeans.fit(data)

    println(label)
    println(kmeans.centroids)

  }
}



