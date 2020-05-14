package com.strings.model.reduce

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, clip, sum}
import breeze.numerics.{exp, log, log2}
import breeze.stats.mean
import com.strings.data.Data
import com.strings.utils.{FileUtils, Utils}
import org.slf4j.LoggerFactory

class TSNE( n_components:Int = 2,
            perplexity:Double = 30.0,
            max_iter:Int = 200,
            learning_rate:Int = 500) {

  private val logger = LoggerFactory.getLogger(classOf[TSNE])

  var initial_monentum:Double = 0.5
  var final_monentum:Double = 0.8
  var min_gain = 0.01
  var tol:Double = 1e-5
  var perplexity_tries:Int = 50
  var n_samples:Int = _
  var n_feature:Int = _

  def l2_distance(X:DenseMatrix[Double]): DenseMatrix[Double] ={
    val data = (0 until X.rows).map(X.t(::,_)).toList
    DenseMatrix(Utils.pair_distance(data):_*)
  }

  def _binary_search(dist:DenseVector[Double],target_entropy:Double): DenseVector[Double] ={
    var precision_min = 0.0
    var precision_max = 1.0e15
    var precision = 1.0e5
    var flag = true
    var beta:DenseVector[Double] = DenseVector.zeros(dist.length)
    for(_ <- 0 until perplexity_tries if flag){
      val exp1 = dist.toArray.filter(_ > 0.0).map(_ * (-1.0) / precision)
      val denom = sum(exp(DenseVector(exp1)))
      beta = exp(dist :*(-1.0/precision)) :/ denom

      val g_beta = DenseVector(beta.toArray.filter(_ > 0.0))
      val entropy = -sum(g_beta :* log2(g_beta))
      val error = entropy - target_entropy

      if(error >0.0){
        precision_max = precision
        precision = (precision + precision_min)/2.0
      }else{
        precision_min = precision
        precision = (precision +precision_max)/2.0
      }
      if(math.abs(error) < tol){
        flag = false
      }
    }
    beta
  }

  def _get_pairwise_affinities(X:DenseMatrix[Double]): DenseMatrix[Double] ={
    val affines = DenseMatrix.zeros[Double](n_samples,n_samples)
    val target_entropy = math.log(perplexity)
    val distance = l2_distance(X)
    val distanceArr = (0 until distance.rows).map(distance.t(::,_)).toList
    for(i <- 0 until n_samples){
      affines(i,::).t := _binary_search(distanceArr(i),target_entropy)
      affines(i,i) = 1.0e-12
    }
    val affines2 = clip(affines,1.0e-100,Double.MaxValue)
    (affines2 :+ affines2.t) :/(2.0 * n_samples)
  }

  def fit_transform(X:DenseMatrix[Double]): DenseMatrix[Double] = {
    n_samples = X.rows
    n_feature = X.cols
    var Y = DenseMatrix.rand[Double](n_samples, n_components)
//    var Y = X.copy
    var velocity = DenseMatrix.zeros[Double](n_samples, n_components)
    var gains = DenseMatrix.ones[Double](n_samples, n_components)

    val P = _get_pairwise_affinities(X)
    var iter_num = 0
    for (_ <- 0 until max_iter) {
      iter_num += 1
      val D = l2_distance(Y)
      val Q = _q_distribution(D)

      val Q_n = Q :/ sum(Q)

      val pmul = if (iter_num < 100) 4.0 else 1.0
      val monentum = if (iter_num < 20) 0.5 else 0.8

      val grads = DenseMatrix.zeros[Double](n_samples, n_components)
      for (i <- 0 until n_samples) {
        val grad1 = (pmul * P(i, ::).t :- Q_n(i, ::).t) :* Q(i, ::).t
        val grad2 = Y(*, ::).map(x => Y(i, ::).t :- x)
        val grad =4.0 * (grad1.t * grad2)
        grads(i,::).t := grad.toDenseVector
      }

      val tmp =(gains :+ 0.2) :* ((grads :> 0.0) :!= (velocity :>0.0)).mapValues(i => if(i) 1.0 else 0.0)
      val tmp2 = (gains :* 0.8) :* ((grads :> 0.0) :== (velocity :>0.0)).mapValues(i => if(i) 1.0 else 0.0)
      //使用下面的方式会出错，检查下什么原因
//      gains := (gains :+ 0.2) :* ((grads :> 0.0) :!= (velocity :>0.0)).mapValues(i => if(i) 1.0 else 0.0) :+
//                    (gains :* 0.8) :* ((grads :> 0.0) :== (velocity :>0.0)).mapValues(i => if(i) 1.0 else 0.0)
      gains = tmp :+ tmp2
      gains = clip(gains,min_gain,Double.MaxValue)
      velocity :=  monentum * velocity - (gains :* grads) * learning_rate.toDouble
      Y := Y :+ velocity
      Y := Y(*,::).map(x => x :- mean(Y,Axis._0).t)
      val error = sum(P :* log(P :/ Q_n))
      logger.info(s"Iteration $iter_num, error $error")
    }
    Y
  }


  def _q_distribution(D: DenseMatrix[Double]): DenseMatrix[Double] = {
    val Q = 1.0 / (D :+ 1.0)
    for (i <- 0 until Q.rows) {
      Q(i, i) = 0.0
    }
    clip(Q, 1.0e-100, Double.MaxValue)
    Q
  }
}

object TSNE{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).toList
    val tsne = new TSNE(max_iter = 100,learning_rate = 100,n_components = 2)
    val W = tsne.fit_transform(DenseMatrix(data:_*))
    val file = "D:\\data\\iris_tsne.txt"
    FileUtils.writeFile(W,file)

  }
}