package com.strings.model.cluster

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, argmax, det, max, norm, pinv, sum}
import com.strings.data.Data
import com.strings.model.metric.Metric
import com.strings.utils.MatrixUtils
import org.slf4j.LoggerFactory

import util.control.Breaks.breakable
import util.control.Breaks.break
import scala.collection.mutable.ArrayBuffer

class GMM(k:Int = 3,
          max_iterations:Int = 2000,
          tolerance:Double = 1e-8) {

  private val logger = LoggerFactory.getLogger(classOf[GMM])

  var means:Array[DenseVector[Double]] = new Array[DenseVector[Double]](k)
  var vars:Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](k)
  var sample_assignments:DenseVector[Int] = _
  var priors:DenseVector[Double] = _
  var responsibility:DenseMatrix[Double] = _
  var responsibilities:ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()

  def _initialize(X:DenseMatrix[Double]): Unit ={
    val n_samples = X.rows
    val X_lst = (0 until n_samples).map(X.t(::,_))
    val rng =  new scala.util.Random()
    priors = DenseVector.ones[Double](k) :/ k.toDouble
    means = rng.shuffle(X_lst).take(k).toArray
//    means = X_lst.take(k).toArray
    for(i <- 0 until k){
      vars(i) = MatrixUtils.calculate_covariance_matrix(X)
    }
  }

  def multivariate_gaussian(X:DenseMatrix[Double],i:Int):DenseVector[Double]={
    val n_features = X.cols
    val mean = means(i)
    val covar = vars(i)
    val determinant = det(covar)
    val likelihoods = DenseVector.zeros[Double](X.rows)
    val X_arr = (0 until X.rows).map(X.t(::,_))

    for((sample,index) <- X_arr.zipWithIndex){
      val n = n_features
      val coeff = 1.0 / (math.pow(2 * Math.PI,n/2) * math.sqrt(determinant))
      val gram = (sample :- mean).t * pinv(covar) * (sample :- mean)
      val exponent = math.exp(-0.5 * gram)
      likelihoods(index) = coeff * exponent
    }
    likelihoods
  }

  def _get_likelihoods(X:DenseMatrix[Double]):DenseMatrix[Double] = {
    val n_samples = X.rows
    val likelihoods = DenseMatrix.zeros[Double](n_samples,k)
    for(i <- 0 until k){
      likelihoods(::,i) := multivariate_gaussian(X,i)
    }
    likelihoods
  }

  def _expectation(X:DenseMatrix[Double]): Unit ={
    val weighted_likelihoods = _get_likelihoods(X)(*,::).map(x => x :* priors)
    val sum_likelihoods = sum(weighted_likelihoods,Axis._1)
    responsibility = weighted_likelihoods(::,*).map(x => x :/ sum_likelihoods) // 列除
    sample_assignments = argmax(responsibility,Axis._1)
    responsibilities.append(max(responsibility,Axis._1))
  }

  def _maximization(X:DenseMatrix[Double]): Unit ={
    for(i <- 0 until k){
      val resp = responsibility(::,i)
      val mean = sum(X(::,*).map(f => resp :* f),Axis._0) :/ sum(resp)
      means(i) = mean.t
      val diff = X(*,::).map(f => f :- mean.t)
      val covariance = diff.t * diff(::,*).map(f => f :* resp) :/sum(resp) // 注意diff(::,*)是取列运算
      vars(i) = covariance
    }
    val n_samples = X.rows
    priors = sum(responsibility,Axis._0).t :/ n_samples.toDouble
  }

  def predict(X:DenseMatrix[Double]): DenseVector[Double] = {
    _initialize(X)
    var iter = 0
    var flag = true
    for (_ <- 0 until max_iterations if flag) {
      iter += 1
      _expectation(X)
      _maximization(X)
      breakable {
        if (responsibilities.length < 2) {
          break()
        }else{
          val n = responsibilities.length
          val diff = norm(responsibilities(n-1) - responsibilities(n-2), 2)
          if (diff <= tolerance) flag = false
        }
    }
  }
    logger.info(s"$iter 之后收敛")
    _expectation(X)
    sample_assignments.map(_.toDouble)
  }

}

object GMM{
  def main(args: Array[String]): Unit = {

    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).toList
    val target = irisData.map(_.apply(4))

    val gmm = new GMM(max_iterations = 100)
    gmm._initialize(DenseMatrix(data:_*))

    val pred = gmm.predict(DenseMatrix(data:_*))
    println(pred)
    val acc =  Metric.accuracy(pred.toArray,target) * 100
    println(f"准确率为: $acc%-5.2f%%")

  }
}
