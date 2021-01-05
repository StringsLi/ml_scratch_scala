package com.strings.model.cluster

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax, min, squaredDistance, sum}
import com.strings.data.Data
import com.strings.utils.MatrixUtils
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.Random

class KmeansPlus (val k:Int = 3,
                  val max_iter:Int = 100,
                  val seed:Long = 1234L,
                  val tolerance: Double = 1e-4){

  private var centroids = mutable.Buffer[DenseVector[Double]]()
  private var cluster = ListBuffer[(Int,DenseVector[Double])]()
  private var label = List[Int]()

  var iterations:Int = 0

  def _init_random_centroids(data : List[DenseVector[Double]]):Unit = {
    val rng  = new Random(seed)
    for(i <- Range(0,k)){
      var index = 0
      if( i == 0) {
        index = rng.nextInt(data.size)
      }else{
        index = _choose_next_center(data)
      }
      centroids.append(data(index))
    }
  }

  def _choose_next_center(data : List[DenseVector[Double]]):Int = {
    val distance:ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()
    for (j <- centroids.indices) {
      val d = data.map(f => MatrixUtils.euclidean_distance(f, centroids(j)))
      distance.append(DenseVector(d.toArray))
    }
   val minDistance = min(DenseMatrix(distance:_*),Axis._0).t
   val square_distance = minDistance.map(x => x * x)
   val probs = square_distance.map(x => x / sum(square_distance))
    argmax(probs)
  }


  def _closest_centroid(centroids:mutable.Buffer[DenseVector[Double]], row:DenseVector[Double]):(Int,DenseVector[Double]) = {
    val distWithIndex =  centroids.zipWithIndex.map(x =>
      (squaredDistance(x._1,row),x._2)
    ).minBy(_._1)
    (distWithIndex._2,row)
  }

  def train(data:List[DenseVector[Double]]):Unit = {
    _init_random_centroids(data)
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

  def _calculate_centroids(cluster:ListBuffer[(Int,DenseVector[Double])]):mutable.Buffer[DenseVector[Double]]= {
    cluster.groupBy(_._1).map { x =>
      val temp = x._2.map(_._2)
      temp.reduce((a, b) => a :+ b).map(_ / temp.length)
    }.toBuffer
  }

  def predictLabel(x:List[DenseVector[Double]]): List[Int] = {
    predict(x).map(_._1)
  }
}

object KmeansPlus{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).map(DenseVector(_)).toList
    val kmeans = new KmeansPlus(max_iter = 100)
    kmeans.train(data)
    println(kmeans.label)

  }
}
