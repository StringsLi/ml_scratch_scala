package com.strings.model.network

import java.util.{ArrayList, List}

import breeze.linalg.*

import breeze.linalg.{DenseMatrix => BDM, max}
import breeze.numerics.exp
import breeze.linalg._


abstract class Layer {
  var layer_type:String = null
  var delta: DenseMatrix[Double] = null
  def prev_delta(delta: BDM[Double]): BDM[Double]
  def forward(forward_data: BDM[Double]): BDM[Double]
  def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double]
  var weights: DenseMatrix[Double] = null
  var moment1: DenseMatrix[Double] = null
  var moment2: DenseMatrix[Double] = null
}

class Sequential() {
  var layers: List[Layer] = new ArrayList[Layer]

  def add(new_layer: Layer): Unit = {
    layers.add(new_layer)
  }
}

class Dense(input_shape: Int, num_hidden: Int) extends Layer {
  val r = scala.util.Random
  this.weights = DenseMatrix.ones[Double](input_shape, num_hidden).map(x => r.nextDouble-0.5) :* .01
  this.moment1 = DenseMatrix.zeros[Double](input_shape, num_hidden)
  this.moment2 = DenseMatrix.zeros[Double](input_shape, num_hidden)
  var hidden_layer: DenseMatrix[Double] = null
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
  var hidden_layer: DenseMatrix[Double] = null
  this.delta = null
  var output_softmax: DenseMatrix[Double] = null

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
      for (i <- 0 to softmax.cols-1){
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
