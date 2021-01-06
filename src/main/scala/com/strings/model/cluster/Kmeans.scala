package com.strings.model.cluster

import breeze.linalg.{DenseVector, squaredDistance}
import com.strings.data.Data

import scala.collection.mutable.ListBuffer
import scala.util.Random

class Kmeans(val k:Int = 3,
             val max_iter:Int = 100,
             val seed:Long = 1234L,
             val tolerance: Double = 1e-4){

  private var centroids = List[DenseVector[Double]]()
  private var cluster = ListBuffer[(Int,DenseVector[Double])]()
  var label = List[Int]()

  var iterations:Int = 0

  def _init_random_centroids(data : List[DenseVector[Double]]):List[DenseVector[Double]] = {
    val rng  = new Random(seed)
    rng.shuffle(data).take(k)
  }

  def _closest_centroid2(centroids:List[DenseVector[Double]],row:DenseVector[Double]):(Int,DenseVector[Double]) = {
        var close_i = 0
        var closest_dist = -1.0
        centroids.zipWithIndex.foreach(centroid => {
          val distance = squaredDistance(centroid._1,row)
          if(closest_dist>distance || closest_dist == -1.0){
            closest_dist = distance
            close_i = centroid._2
          }
        })
    (close_i,row)
  }

  def _closest_centroid(centroids:List[DenseVector[Double]],row:DenseVector[Double]):(Int,DenseVector[Double]) = {
      val distWithIndex =  centroids.zipWithIndex.map(x =>
                          (squaredDistance(x._1,row),x._2)
                          ).minBy(_._1)
      (distWithIndex._2,row)
  }

  def train(data:List[DenseVector[Double]]):Unit = {
    centroids = _init_random_centroids(data)
    var flag = true
    for(_ <- Range(0,max_iter) if flag){
      iterations += 1
      data.foreach{d =>
        val b = _closest_centroid(centroids, d)
        cluster.append(b)
      }
      val prev_centroid = centroids
      centroids = _calculate_centroids(cluster)
      cluster = ListBuffer[(Int,DenseVector[Double])]()
      val diff = prev_centroid.zip(centroids).map(x => squaredDistance(x._2,x._1))
      if( diff.sum < tolerance){
        flag = false
      }
    }
    label = predictLabel(data)
  }

  def predict(data:List[DenseVector[Double]]):List[(Int,DenseVector[Double])]= {
    data.map(x => _closest_centroid(centroids,x))
  }

  def _calculate_centroids(cluster:ListBuffer[(Int,DenseVector[Double])]):List[DenseVector[Double]]= {
    cluster.groupBy(_._1).map { x =>
      val temp = x._2.map(_._2)
      temp.reduce((a, b) => a :+ b).map(_ / temp.length)
    }.toList
  }

  def predictLabel(x:List[DenseVector[Double]]): List[Int] = {
    predict(x).map(_._1)
  }
}

object Kmeans{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val kmeans = new Kmeans(max_iter = 100)
    kmeans.train(data)
    println(kmeans.label)

  }
}
