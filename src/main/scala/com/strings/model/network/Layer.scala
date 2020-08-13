package com.strings.model.network

import java.util

import breeze.linalg.*
import breeze.linalg.{max, DenseMatrix => BDM}
import breeze.numerics.exp
import breeze.linalg._

import scala.util.Random


abstract class Layer {
  var layer_type:String = _
  var delta: DenseMatrix[Double] = _
  def prev_delta(delta: BDM[Double]): BDM[Double]
  def forward(forward_data: BDM[Double]): BDM[Double]
  def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double]
  var weights: DenseMatrix[Double] = _
  var moment1: DenseMatrix[Double] = _
  var moment2: DenseMatrix[Double] = _
}

class Sequential() {
  var layers: util.List[Layer] = new util.ArrayList[Layer]()

  def add(new_layer: Layer): Unit = {
    layers.add(new_layer)
  }
}

class Dense(input_shape: Int, num_hidden: Int) extends Layer {
  val r: Random.type = scala.util.Random
  this.weights = DenseMatrix.ones[Double](input_shape, num_hidden).map(_ => r.nextDouble-0.5) :* .01
  this.moment1 = DenseMatrix.zeros[Double](input_shape, num_hidden)
  this.moment2 = DenseMatrix.zeros[Double](input_shape, num_hidden)
  var hidden_layer: DenseMatrix[Double] = _
  this.delta = null
  this.layer_type = "Dense"

  def get_num_hidden: Int = num_hidden
  def get_input_shape: Int = input_shape

  override def forward(forward_data: BDM[Double]): BDM[Double] = {
    hidden_layer = forward_data * weights
    hidden_layer
  }

  override def prev_delta(delta: BDM[Double]): BDM[Double] = {
    delta * weights.t
  }

  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    backward_data.t * delta
  }
}


class Activation(var kind: String) extends Layer {
  this.layer_type = "Activation"
  var hidden_layer: DenseMatrix[Double] = _
  this.delta = null
  var output_softmax: DenseMatrix[Double] = _

  override def forward(input_data: DenseMatrix[Double]) : DenseMatrix[Double] = {

    if (kind == "relu") {
      hidden_layer = input_data.map(x => max(0.0, x))
      hidden_layer
    }

    else if (kind == "linear") {
      hidden_layer = input_data
      hidden_layer
    }

    else if (kind == "softmax") {
      val softmax = exp(input_data)
      val divisor = breeze.linalg.sum(softmax(*, ::))
      for (i <- 0 until  softmax.cols){
        softmax(::, i) := softmax(::, i) :/ divisor
      }
      softmax
    }
    else {
      println("MAJOR ERROR1")
      input_data
    }
  }

  def relu_grad(value: Double):Double = {if (value <= 0) {0} else {1}}

  override def prev_delta(delta: DenseMatrix[Double]): DenseMatrix[Double] = {
    if (kind == "relu") {
      delta :* hidden_layer.map(relu_grad)
      delta
    }
    else if (kind == "linear") {
      delta
    }
    else {
      println("MAJOR ERROR2")
      delta
    }
  }

  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    println("MAJOR ERROR - ACTIVATION LAYER SHOULD NOT COMPUTE GRADIENT")
    backward_data
  }
}
