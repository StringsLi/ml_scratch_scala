package com.strings.model.cluster

import breeze.linalg.{DenseVector, squaredDistance}
import com.strings.data.Data

import scala.collection.mutable.ListBuffer
import scala.util.Random

class KMedoid(val k:Int = 3,
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

  def _closest_centroid(centroids:List[DenseVector[Double]],row:DenseVector[Double]):(Int,DenseVector[Double]) = {
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

  // 要不同的是k-means聚类算法更新聚簇中心的时候直
  //接计算的均值，而k-mediods聚类算法更新聚簇中心的时候先对每个聚簇中心计算每一个点到簇内其他点的距离之和，然
  //后再选择距离最小的点来作为新的聚簇中心
  def _calculate_centroids(clusters:ListBuffer[(Int,DenseVector[Double])]):List[DenseVector[Double]]= {
    val centorid:Array[DenseVector[Double]] = new Array[DenseVector[Double]](k)
    cluster.groupBy(_._1).foreach { x =>
      val curr_cluster = x._2.map(_._2)
      var min_dist = Double.MaxValue
      for (point <- curr_cluster) {
        val dist = curr_cluster.map(t => squaredDistance(t, point))
        val total_dist = dist.sum
        if (total_dist < min_dist) {
          centorid(x._1) = point
          min_dist = total_dist
        }
      }
    }
    centorid.toList
  }

  def predictLabel(x:List[DenseVector[Double]]): List[Int] = {
    predict(x).map(_._1)
  }
}


object KMedoid{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val kmeans = new KMedoid(max_iter = 100)
    kmeans.train(data)
    println(kmeans.label)

  }
}


