package com.strings.utils

import breeze.linalg.{DenseMatrix, DenseVector, squaredDistance}

object Utils {

  def log2(x: Double):Double = Math.log(x) / Math.log(2)

  //Calculate the entropy of label array y
  def calculate_entropy(y:Array[Double]):Double ={
    val unique_labels = y.toSet.toArray
    var entropy:Double = 0.0
    for (label <-  unique_labels) {
      val count:Int = y.filter(_ ==label).length
      val p:Double = count / y.length.toDouble
      entropy += -p * log2(p)
    }
    entropy
  }

  def calculate_entroy2(y:Array[Double]):Double = {
      y.groupBy(x => x)
       .mapValues(x => x.length.toDouble / y.length)
       .mapValues(p => -p * log2(p))
       .foldLeft(0.0)(_ + _._2)
  }

  def caclulate_gini(y:Array[Double]):Double = {
      1 - y.groupBy(x => x).
        mapValues(x => x.length.toDouble / y.length.toDouble).
        mapValues(p => p * p).
        foldLeft(0.0)(_ + _._2)
  }


  def calculate_variance(y:Array[Double]):Double = {
    val size = y.length
    val mean = y.sum / size
    y.map(x => (x - mean) * (x - mean)).sum / size
  }

  def clip(value: Double):Double = if (value < 0.00001) 0.00001
        else if (value > 0.99999) 0.99999
        else value

  def one_hot(y: Array[Double]): Array[Array[Double]] = {
    val n_samples = y.length
    val n_classes = y.toSet.size
    val one_hot = Array.ofDim[Double](n_samples, n_classes)
    for (i <- 0 until  n_samples) {
      one_hot(i)(y(i).toInt) = 1.0
    }
    one_hot
  }

  def split(y:Array[Array[Double]]):(Array[Double],Array[Double]) = {
    (y.map(_(0)),y.map(_(1)))
  }

  def pair_distance(X:List[DenseVector[Double]]):Array[Array[Double]] = {
    val nSample = X.length
    val pairDist:Array[Array[Double]] = Array.ofDim[Double](nSample,nSample)
    for(i <- 0 until nSample){
      for(j <- 0 until nSample){
        pairDist(i)(j) = squaredDistance(X(i),X(j))
      }
    }
    pairDist
  }


  def rowsToMatrix(in: TraversableOnce[DenseVector[Double]]): DenseMatrix[Double] = {
    rowsToMatrix(in.toArray)
  }

  def rowsToMatrix(inArr: Array[DenseVector[Double]]): DenseMatrix[Double] = {
    val nRows = inArr.length
    val nCols = inArr(0).length
    val outArr = new Array[Double](nRows * nCols)
    var i = 0
    while (i < nRows) {
      var j = 0
      val row = inArr(i)
      while (j < nCols) {
        outArr(i + nRows * j) = row(j)
        j = j + 1
      }
      i = i + 1
    }
    val outMat = new DenseMatrix[Double](nRows, nCols, outArr)
    outMat
  }



}
