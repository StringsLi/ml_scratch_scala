package com.strings.model.cluster
import util.control.Breaks._
import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.{exp, pow, sqrt}
import com.strings.data.Data

class MeanShift(val kernel_bandwidth:Double) {

  def gaussian_kernel(distance:DenseMatrix[Double],kernel_bandwidth:Double) ={
    val euclidean_distance = sqrt(sum(pow(distance,2),Axis._1))
    val xishu = 1.0 /kernel_bandwidth*math.sqrt(2* 3.14)
    val value = xishu * exp(pow(euclidean_distance / kernel_bandwidth,2) * (-0.5))
    value
  }


  def shiftPoint(point:DenseVector[Double],points:List[DenseVector[Double]],kernel_bandwidth:Double) ={
    val ll = points.map(t => point - t)
    val jj = gaussian_kernel(DenseMatrix(ll:_*),kernel_bandwidth)
    jj
  }

  def fit(points:List[DenseVector[Double]])={
    val shift_points = points
    val max_min_dist = 1
    var iteration_number = 0
    val still_shifting:Array[Boolean] = Array.fill(points.length)(true)

    val MIN_DISTANCE = 1e-6
    while (max_min_dist > MIN_DISTANCE) {
      val max_min_dist = 0.0
      iteration_number += 1

      for (i <- 0 until shift_points.length) {
        breakable {
          if (!still_shifting(i)) {
            break
          }
        }
        var p_new = shift_points(i)
        var p_new_start = p_new
        shiftPoint(p_new, points, kernel_bandwidth)
      }
    }

  }

}

object MeanShift{

  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val ms = new MeanShift(kernel_bandwidth = 0.3)
    ms.fit(data)
//    println(kmeans.label)

  }

}
