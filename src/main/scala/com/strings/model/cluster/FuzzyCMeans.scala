package com.strings.model.cluster

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, argmax, norm, squaredDistance, sum}
import breeze.numerics.pow
import breeze.stats.distributions.Rand
import com.strings.data.Data

import scala.util.Random
import scala.util.control.Breaks

class FuzzyCMeans(val k:Int = 4,
                  val fuzziness:Int = 2,
                  val epsilon: Double = 1e-8,
                  val seed:Long = 1234L,
                  val max_iter: Int = 10000){

  private var data:List[DenseVector[Double]] = _
  private var fuzzyMatrix:DenseMatrix[Double] = _
  private var centrids:List[DenseVector[Double]] = _
  private var iter_c:Int = _

  def _init_random_centroids(data : List[DenseVector[Double]]):List[DenseVector[Double]] = {
    val rng  = new Random(seed)
    rng.shuffle(data).take(k)
  }

  def euclidean_distance(x1:DenseVector[Double],x2:DenseVector[Double]): Double ={
    math.sqrt(sum((x1 :- x2) :* (x1 :- x2)))
  }

  def fit(X:List[DenseVector[Double]])= {
    data = X
    fuzzyMatrix = DenseMatrix.rand(data.length, k, rand = Rand.uniform)
    fuzzyMatrix = fuzzyMatrix(::, *) / sum(fuzzyMatrix, Axis._1)

    println(fuzzyMatrix)

    centrids = _init_random_centroids(data)
    var fuzzy_matrix_powerd = pow(fuzzyMatrix, fuzziness)

    var sse_error = 0.0

    for (j <- 0 until k) {
      for (i <- 0 until data.length) {
        sse_error = sse_error + fuzzy_matrix_powerd(i, j) * math.pow(norm(data(i) - centrids(j)), 2)
      }
    }
    var iter_count = 1

    Breaks.breakable {
      for (_ <- Range(0, max_iter)) {
        iter_count += 1

        fuzzy_matrix_powerd = pow(fuzzyMatrix, fuzziness)

        fuzzy_matrix_powerd = fuzzy_matrix_powerd(::, *) / sum(fuzzy_matrix_powerd, Axis._1)

        val new_centroid = fuzzy_matrix_powerd.t * DenseMatrix(data: _*)

        var new_fuzzy_matrix = DenseMatrix.zeros[Double](data.length, k)

        for (i <- 0 until new_fuzzy_matrix.rows) {
          for (j <- 0 until new_fuzzy_matrix.cols) {
            val centroid_j = (0 until k).map(new_centroid.t(::, _))
            new_fuzzy_matrix(i, j) = 1.0 / norm(data(i) - centroid_j.apply(j))
          }
        }

        new_fuzzy_matrix = pow(new_fuzzy_matrix, 2.0 / (fuzziness - 1))
        new_fuzzy_matrix = fuzzy_matrix_powerd(::, *) / sum(fuzzy_matrix_powerd, Axis._1)
        val new_fuzzy_matrix_powered = pow(new_fuzzy_matrix, fuzziness)
        var new_sse_err = 0.0

        for (j <- 0 until k) {
          for (i <- 0 until data.length) {
            new_sse_err += new_fuzzy_matrix_powered(i, j) * math.pow(norm(data(i) - centrids(j)), 2)
          }
        }
        if (math.abs(sse_error - new_sse_err) < epsilon){
          Breaks.break()
        }
        centrids = (0 until k).map(new_centroid.t(::, _)).toList
        fuzzyMatrix = new_fuzzy_matrix

      }
    }

    iter_c = iter_count
    argmax(fuzzyMatrix,Axis._1)
  }



}

object FuzzyCMeans{
  def main(args: Array[String]): Unit = {

    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val kmeans = new FuzzyCMeans(k = 3)
    val label = kmeans.fit(data)
    println(label)
    println(kmeans.iter_c)

  }
}
