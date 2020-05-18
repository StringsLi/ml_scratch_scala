package com.strings.model.cluster

import breeze.linalg.{DenseVector, sum}

class LVQ(k:Double = 4,
          lr:Double = 0.01,
          nums_iters:Int = 100,
          delta:Double = 0.001) {

  def euclidean_distance(x1:DenseVector[Double],x2:DenseVector[Double]): Double ={
    math.sqrt(sum((x1 :- x2) :* (x1 :- x2)))
  }

  def 

}
