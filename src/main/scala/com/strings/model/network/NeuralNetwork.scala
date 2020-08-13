package com.strings.model.network

import java.io.File
import java.util.{ArrayList, List}

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax, convert, csvread}
import breeze.numerics.sqrt

class NeuralNetwork(lr:Double = .001,
                    num_iters:Int = 1000,
                    optimizer:String = "adam",
                    s:Double = .9,
                    batch_size:Int = 16,
                    r:Double = .999) {

  class Sequence {
    var layers: List[Layer] = new ArrayList[Layer]

    def add(new_layer: Layer): Unit = {
      layers.add(new_layer)
    }
  }

  def evaluate(s: Sequential, x:DenseMatrix[Double], y:DenseVector[Double]):Double = {
    var f = x
    // Forward
    for (layer <- 0 to s.layers.size-1) {
      f = s.layers.get(layer).forward(f)
    }
    val softmax = f
    val predictions = argmax(softmax, Axis._1)
    val diff = predictions-convert(y, Int)
    var count = 0
    for (i <- 0 to diff.size-1){
      if (diff(i) == 0) {count += 1}
      else {}
    }
    count.toDouble/diff.size.toDouble
  }

  def one_hot(y: DenseVector[Double],n_classses:Int): DenseMatrix[Double] = {
    val n_samples = y.length
    val one_hot = Array.ofDim[Double](n_samples, n_classses)
    for (i <- 0 to n_samples - 1) {
      one_hot(i)(y(i).toInt) = 1.0
    }
    DenseMatrix(one_hot: _*)
  }

  def get_batch(x_train:DenseMatrix[Double], y_train:DenseVector[Int], batch_size:Int): (DenseMatrix[Double], DenseVector[Int]) = {
    val rand = scala.util.Random
    val x_batch = DenseMatrix.zeros[Double](batch_size, x_train.cols)
    val y_batch = DenseVector.zeros[Int](batch_size)

    for (i <- 0 until batch_size){
      val batch_index = rand.nextInt(x_train.rows-batch_size)
      x_batch(i, ::) := x_train(batch_index, ::)
      y_batch(i) = y_train(batch_index)
    }
    (x_batch, y_batch)
  }

  def fit(seq: Sequential, x: DenseMatrix[Double], y: DenseVector[Double]): Sequential = {

    val x_train = x
    val y_train = convert(y, Int)

    val class_count = seq.layers.get(seq.layers.size-2).asInstanceOf[Dense].get_num_hidden

    val numerical_stability = .00000001

    for (iterations <- 0 to num_iters) {

      val (x_batch, y_batch) = get_batch(x_train, y_train, batch_size)

      var f = x_batch
      // Forward
      for (layer <- 0 until seq.layers.size) {
        f = seq.layers.get(layer).forward(f)
      }
      var softmax = f

      val y_one_hot = one_hot(convert(y_batch,Double),class_count)

      softmax = softmax - y_one_hot
      seq.layers.get(seq.layers.size-1).delta = softmax :/ batch_size.toDouble

      // Compute Errors
      for (i <- seq.layers.size-2 to 0 by -1) {
        seq.layers.get(i).delta = seq.layers.get(i).prev_delta(seq.layers.get(i+1).delta)
      }

      // Compute and Update Gradients
      for (i <- seq.layers.size-2 to 0 by -1) {
        if (seq.layers.get(i).layer_type == "Dense") {
          val gradient =
            if (i == 0) {
              seq.layers.get(i).asInstanceOf[Dense].compute_gradient(x_batch, seq.layers.get(i+1).delta)
            } else {
              seq.layers.get(i).asInstanceOf[Dense].compute_gradient(seq.layers.get(i-1).asInstanceOf[Activation].hidden_layer, seq.layers.get(i+1).delta)
            }

          val layer = seq.layers.get(i)

          if (optimizer == "sgd") {
            layer.weights -= lr * gradient
          }
          else if (optimizer == "momentum") {
            layer.moment1 = s * layer.moment1 + lr * gradient
            layer.weights -= layer.moment1
          }
          else if (optimizer == "adam") {
            layer.moment1 = s * layer.moment1 + (1-s) * gradient
            layer.moment2 = r * layer.moment2 + (1-r) * (gradient :* gradient)
            val m1_unbiased = layer.moment1 :/ (1-scala.math.pow(s, iterations+1))
            val m2_unbiased = layer.moment2 :/ (1-scala.math.pow(r, iterations+1))
            layer.weights -= lr * m1_unbiased :/ (sqrt(m2_unbiased) + numerical_stability)
          }
        }
      }
      if (iterations % 10 == 0) {
        println("Iterations: " + iterations + " Training Acc: " + evaluate(seq, x(0 until 1000, ::), y(0 until 1000)))
      }
    }
    seq
  }

}

object NeuralNetwork{
  def main(args: Array[String]): Unit = {
    val jf = new File("E:\\gitlab\\Strings_Spark_Utils\\src\\main\\resources/mnist_small.csv")
    val data = csvread(jf)
    val features = data(::, 1 to 784)
    val labels = data(::, 0)

    val t = new Sequential()
    t.add(new Dense(784, 100))
    t.add(new Activation("relu"))
    t.add(new Dense(100, 10))
    t.add(new Activation("softmax"))
    val neuralNetwork = new NeuralNetwork()
    neuralNetwork.fit(t,features,labels)

  }
}
