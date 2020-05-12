package com.strings.model.classification

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.{exp, log, pow, sqrt}
import breeze.stats.{mean, variance}
import com.strings.data.Data
import com.strings.model.metric.Metric
import com.strings.model.network.SoftMax

class NaiveBayesClassifier(n_classes:Int = 2) extends ClassificationModel {

  var _mean:DenseMatrix[Double] = _
  var _var:DenseMatrix[Double] = _
  var _priors:DenseVector[Double] = _
  var n_features:Int = _

  override def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    val prob = _predict(x)
    argmax(prob,Axis._1).map(_.toDouble)
  }

  def fit(X:DenseMatrix[Double],y:DenseVector[Double]):Unit = {
    val y_dist = y.toArray.toSet
    require(y_dist.equals(Set(1.0,0.0)))

    n_features = X.cols
    _mean = DenseMatrix.zeros(n_classes,n_features)
    _var = DenseMatrix.zeros(n_classes,n_features)
    _priors = DenseVector.zeros(n_classes)

    for(c <- 0 until n_classes){
      val X_c = y.toArray.indices.filter(i => y(i) == c).map(X.t(::,_)).map(_.toDenseMatrix)
      val X_cc = DenseMatrix.vertcat(X_c:_*)
      _mean(c,::) := mean(X_cc(::,*))
      _var(c,::) := variance(X_cc(::,*))
      _priors(c) = X_cc.rows /X.rows.toDouble
    }
  }

  def _predict(X:DenseMatrix[Double]):DenseMatrix[Double] = {
    val prediction = (0 until X.rows).map(i => X.t(::,i)).map(j => _predict_row(j))
    SoftMax.value(DenseMatrix(prediction:_*))
  }

  def _predict_row(x:DenseVector[Double]):Array[Double] = {
    val output = new Array[Double](n_classes)
      for(y <- 0 until n_classes){
        val prior = math.log(_priors(y))
        val posterior = sum(log(_pdf(y,x)))
        output(y) = prior + posterior
      }
    output
  }

  def _pdf(n_class:Int,x:DenseVector[Double]):DenseVector[Double] = {
    val mean = _mean(n_class,::).t
    val variance = _var(n_class,::).t
    val tmp = pow(x - mean,2) :/ variance.map(_*2)
    val numerator:DenseVector[Double] = exp(-1.0 * tmp)
    val denominator = sqrt(2 * 3.14 * variance)
     numerator / denominator
  }

}


object NaiveBayesClassifier{
  def main(args: Array[String]): Unit = {

    val iris_data = Data.iris4BinaryClassification()
    val train_test_data = Data.train_test_split(iris_data._1,iris_data._2,0.32,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    val nb = new NaiveBayesClassifier()
    nb.fit(DenseMatrix(trainX:_*),DenseVector(trainY))
    val pred1 = nb.predict(DenseMatrix(testX:_*))
    val acc =  Metric.accuracy(pred1.toArray,testY) * 100

    println(f"准确率为: $acc%-5.2f%%")

  }
}
