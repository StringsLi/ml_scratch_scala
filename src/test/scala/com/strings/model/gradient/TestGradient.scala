package com.strings.model.gradient

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}


object TestGradient {

  def genData(num_examples:Int = 10000,num_inputs:Int = 2):(BDM[Double],BDV[Double]) = {
    val x_train: BDM[Double] = BDM.rand(num_examples, num_inputs)
    val ones = BDM.ones[Double](num_examples, 1)
    val x_cat = BDM.horzcat(ones, x_train)
    val nos = BDV.rand(num_examples) :* 0.01
    val y_train = x_cat * BDV(2.3, 6.4, -3.2) + nos
    (x_train,y_train)
  }

  def main(args: Array[String]): Unit = {
    val sgd = new StochasticGradientDescent(num_iters = 1000)
    val batch = new BatchGradientDescent(num_iters = 1000)
    val miniBatch = new MiniBatchGradientDescent(num_iters = 1000)
    val momentumGradientDescent = new MomentumGradientDescent(num_iters = 1000)
    val adagrad = new AdaptiveGradientDescent(num_iters = 1000)
    val rmsprop = new RMSProp(num_iters = 1000)
    val nesterovAccelerateGradient = new NesterovAccelerateGradient(num_iters = 1000)
    val adam = new AdaptiveMomentEstimation(num_iters = 1000)
    val adamax = new AdaMax(num_iters = 1000)
    val nadam = new Nadam(num_iters = 1000)
    val amsgrad = new AMSGrad(num_iters = 1000)

    val gradients:Array[BaseGradient] = Array(sgd,batch,miniBatch,momentumGradientDescent,adagrad,
                          rmsprop,rmsprop,nesterovAccelerateGradient,adam,
                          adamax,nadam,amsgrad)

    val (x_train,y_train) = genData()

    println("     模型名称                  迭代次数                权重            ")
    gradients.foreach { model =>
      val weights = model.fit(x_train, y_train)
      val model_name =  model.getClass.getName.split("\\.")(4)
      println(f"$model_name%-10s  ${weights._2}%10d ${weights._1.toArray.mkString(",")}%-20s")
    }



  }

}

