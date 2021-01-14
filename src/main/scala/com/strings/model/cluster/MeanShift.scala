package com.strings.model.cluster
import util.control.Breaks._
import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum, tile}
import breeze.numerics.{exp, pow, sqrt}
import com.strings.data.Data
import scala.collection.mutable.ArrayBuffer

class MeanShift(val kernel_bandwidth:Double) {

  def euclidean_distance(x1:DenseVector[Double],x2:DenseVector[Double]): Double ={
    math.sqrt(sum((x1 :- x2) :* (x1 :- x2)))
  }

  def gaussian_kernel(distance:DenseMatrix[Double],kernel_bandwidth:Double):DenseVector[Double] ={
    val euclidean_distance = sqrt(sum(pow(distance,2),Axis._1))
    val coefficient = 1.0 / (kernel_bandwidth*math.sqrt(2* math.Pi))
    coefficient * exp(pow(euclidean_distance / kernel_bandwidth,2) * (-0.5))
  }


  def shiftPoint(point:DenseVector[Double],points:List[DenseVector[Double]],kernel_bandwidth:Double):DenseVector[Double] ={
    val distance = points.map(t => point - t)
    val point_weights = gaussian_kernel(DenseMatrix(distance:_*),kernel_bandwidth)
    val tiled_weight = tile(point_weights,1,point.length)
    val denominator = sum(point_weights)
    sum(tiled_weight :* DenseMatrix(points:_*),Axis._0).t / denominator //各个元素对应相乘
  }

  def group_points(points:List[DenseVector[Double]]):List[Int] = {
    val cluster_ids:ArrayBuffer[Int] = new ArrayBuffer[Int]()
    var cluster_idx = 0
    val cluster_centers:ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()

    for((point,i) <- points.zipWithIndex){
      if(cluster_ids.length == 0){
        cluster_ids.append(cluster_idx)
        cluster_centers.append(point)
        cluster_idx += 1
      }else{
        for((center,j) <- cluster_centers.zipWithIndex){
          val dist = euclidean_distance(point, center)
          if(dist < 0.1){
            cluster_ids.append(j)
          }
        }

        if(cluster_ids.length < i + 1) {
          cluster_ids.append(cluster_idx)
          cluster_centers.append(point)
          cluster_idx += 1
        }
      }
    }
    cluster_ids.toList
  }

  def fit(points:List[DenseVector[Double]]):List[Int]={
    val shift_points = points
    var max_min_dist = 1.0
    var iteration_number = 0
    val still_shifting:Array[Boolean] = Array.fill(points.length)(true)
    val MIN_DISTANCE = 1e-5
    while (max_min_dist > MIN_DISTANCE) {
      max_min_dist = 0.0
      iteration_number += 1

      for (i <- 0 until shift_points.length) {
        breakable {
          if (!still_shifting(i)) {
            break
          }
        }
        var p_new = shift_points(i).copy
        val p_new_start = p_new.copy
        p_new = shiftPoint(p_new, points, kernel_bandwidth)
        val dist = euclidean_distance(p_new, p_new_start)
        if(dist > max_min_dist){
          max_min_dist = dist
        }
        if(dist < MIN_DISTANCE){
          still_shifting(i) = false
        }
        shift_points(i) := p_new
      }
    }
    group_points(shift_points)
  }

}

object MeanShift{

  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val ms = new MeanShift(kernel_bandwidth = 0.3)
    val label = ms.fit(data)
    println(label)
  }

}
