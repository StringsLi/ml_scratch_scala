package com.strings.model.network

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.{pow, sigmoid}
import com.strings.data.{Data, Dataset}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

class RBM(n_hidden:Int=128, var learning_rate:Double=0.1, batch_size:Int=40, max_epochs:Int=100) {

  private val logging = LoggerFactory.getLogger(classOf[RBM])

  var W:DenseMatrix[Double] = _
  var bias_v:DenseVector[Double] = _
  var bias_h:DenseVector[Double] = _
  var errors: ArrayBuffer[Double] = new ArrayBuffer[Double]()
  var n_visible:Int = _
  var y:DenseVector[Double] = _

  def fit(X:DenseMatrix[Double]): Unit ={
    n_visible = X.cols
    _init_weights()
    train(X)
  }

  def _init_weights(): Unit ={
    W = DenseMatrix.rand[Double](n_visible,n_hidden)
    bias_v = DenseVector.zeros[Double](n_visible)
    bias_h = DenseVector.zeros[Double](n_hidden)
  }

  def _sample(X:DenseMatrix[Double]):DenseMatrix[Double] = {
    val bol = X :> DenseMatrix.rand[Double](X.rows,X.cols)
    bol.map(x => if(x) 1.0 else 0.0)
  }

  def train(X:DenseMatrix[Double]): Unit ={
    y = DenseVector.zeros[Double](X.rows)
    val dataset = new Dataset(X,y)
    for(i <- 0 until max_epochs){
      var error = 0.0
      val batch_data = dataset.get_minibatch(batch_size,doShffule = false)

      for((batch,_) <- batch_data){
        val forward = batch * W
        val postive_hidden = sigmoid(forward(*,::).map(_ :+ bias_h))
        val hidden_states = _sample(postive_hidden)
        val postive_associations = batch.t * postive_hidden


        val negative1 = hidden_states * W.t
        var negative_visible = sigmoid(negative1(*,::).map(_ :+ bias_v))
        negative_visible = _sample(negative_visible)
        val forword2 = negative_visible * W
        val negative_hidden = forword2(*,::).map(_ :+ bias_h)
        val negative_associations = negative_visible.t * negative_hidden

        learning_rate = learning_rate / batch.rows.toDouble
        W :+= (postive_associations :- negative_associations) :* (learning_rate /batch_size)
        bias_h :+= (sum(negative_hidden,Axis._0).t :- sum(negative_associations,Axis._0).t) :* learning_rate
        bias_v :+= (sum(batch,Axis._0).t :- sum(negative_visible,Axis._0).t) :* learning_rate

        error += sum(pow(batch :- negative_visible,2))

      }
      errors.append(error)
      println(s"Iteration $i, error $error" )
    }
    logging.debug("Weights: %s".format(W))
    logging.debug("Hidden bias: %s".format(bias_h))
    logging.debug("Visible bias: %s".format(bias_v))
  }

  def predict(X:DenseMatrix[Double]):DenseMatrix[Double] = {
    val forward = X * W
    sigmoid(forward(*,::).map(_ :+ bias_h))
  }

}

object RBM{
  def main(args: Array[String]): Unit = {
    val irisData = Data.irisData
    val data = irisData.map(_.slice(0,4)).toList
    val target = irisData.map(_.apply(4))

    val X = DenseMatrix(data:_*)
    val rbm = new RBM()
    rbm.fit(X)

    println(rbm.predict(X))

  }
}
