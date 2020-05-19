package com.strings.model.cluster

import breeze.linalg.{DenseMatrix, DenseVector, argmin, sum}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class LVQ(t:Array[Int],
          lr:Double = 0.1,
          nums_iters:Int = 400) {

  val c = t.distinct.length
  val q = t.length
  var C: Map[Int, ArrayBuffer[Int]] = Map()
  var p: DenseMatrix[Double] = _
  var labels: DenseVector[Int] = _

  def euclidean_distance(x1: DenseVector[Double], x2: DenseVector[Double]): Double = {
    require(x1.length == x2.length)
    math.sqrt(sum((x1 :- x2) :* (x1 :- x2)))
  }

  def fit(X: DenseMatrix[Double], y: DenseVector[Int]) = {
    p = DenseMatrix.zeros[Double](q, X.cols)
    for (i <- 0 until q) {
      C += (i -> ArrayBuffer[Int]())
      val candidate_indices = y.toArray.indices.filter(f => y(f) == t(i))
      val target_indice = Random.shuffle(candidate_indices.toList).take(1).apply(0)
      p(i, ::) := X(target_indice, ::)
    }


    var p_arr = (0 until p.rows).map(p.t(::, _))
    for (_ <- 0 until nums_iters) {
      val j = Random.shuffle(Range(0, y.length).toList).take(1).apply(0)
      val x_j = X(j, ::).t
      val d = p_arr.map(f => euclidean_distance(f, x_j))
      val idx: Int = argmin(d.toArray)
      if (y(j) == t(idx)) {
        p(idx, ::) := p(idx, ::) :+ ((X(j, ::) :- p(idx, ::)) :* lr)  // :+ 和 :* 运算优先级一致
      } else {
        p(idx, ::) := p(idx, ::) :- ((X(j, ::) :- p(idx, ::)) :* lr)
      }

    }
    p_arr = (0 until p.rows).map(p.t(::, _))
    for (j <- 0 until X.rows) {
      val d = p_arr.map(f => euclidean_distance(f, X(j, ::).t))
      val idx: Int = argmin(DenseVector(d.toArray))
      C(idx).append(j)
    }

    labels = DenseVector.zeros[Int](X.rows)
    for (i <- 0 until q) {
      for (j <- C(i)) {
        labels(j) = i
      }
    }
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Int] = {
    val p_arr = (0 until p.rows).map(p.t(::, _))
    val preds_y: ArrayBuffer[Int] = new ArrayBuffer[Int]()
    for (j <- 0 until X.rows) {
      val d = p_arr.map(f => euclidean_distance(f, X(j, ::).t))
      val idx: Int = argmin(DenseVector(d.toArray))
      preds_y.append(t(idx))
    }
    DenseVector(preds_y.toArray)
  }
}

object LVQ{
  def main(args: Array[String]): Unit = {

    val X = Array(Array(0.697,0.460),Array(0.774,0.376),Array(0.634,0.264),Array(0.608,0.318),Array(0.556,0.215),
                  Array(0.403,0.237),Array(0.481,0.149),Array(0.437,0.211),Array(0.666,0.091),Array(0.243,0.267),
                  Array(0.245,0.057),Array(0.343,0.099),Array(0.639,0.161),Array(0.657,0.198),Array(0.360,0.370),
                  Array(0.593,0.042),Array(0.719,0.103),Array(0.359,0.188),Array(0.339,0.241),Array(0.282,0.257),
                  Array(0.748,0.232),Array(0.714,0.346),Array(0.483,0.312),Array(0.478,0.437),Array(0.525,0.369),
                  Array(0.751,0.489),Array(0.532,0.472),Array(0.473,0.376),Array(0.725,0.445),Array(0.446,0.459))

   val XX = DenseMatrix(X:_*)
   val y = DenseVector.zeros[Int](XX.rows)

    for(i <- 9 until 21){
      y(i) = 1
    }

    val t = Array(0,1,1,0,0)
    println(y)
    val lvq = new LVQ(t)
    lvq.fit(XX,y)

    println(lvq.C)
    println(lvq.labels)
    println(lvq.predict(XX))

  }
}



