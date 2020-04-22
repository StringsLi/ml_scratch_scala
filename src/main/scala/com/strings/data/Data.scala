package com.strings.data

object Data {

  val irisData = scala.io.Source.fromFile("D:/data/iris.csv")
            .getLines().toSeq
            .map{_.split(",")
            .filter(_.length() > 0)
            .map(_.toDouble)}
            .toArray

  /**
   * @return X,y
   */
  def iris4BinaryClassification():(Array[Array[Double]],Array[Double]) = {
    val data = irisData.map(x => (x.apply(4),x.slice(0,4))).slice(0,99)
    (data.map(_._2),data.map(_._1))
  }

  def iris4MutilClassification():(Array[Array[Double]],Array[Double]) = {
    val data = irisData.map(x => (x.apply(4),x.slice(0,4)))
    (data.map(_._2),data.map(_._1))
  }

  def iris4Regression():(Array[Array[Double]],Array[Double]) ={
    val data = irisData.map(x => (x.apply(3), x.slice(0, 3)))
    (data.map(_._2),data.map(_._1))
  }

  def train_test_split(X:Array[Array[Double]],y:Array[Double], test_size:Double = 0.5,
                        shuffle:Boolean = true,seed:Long = 1234L)={
    require(test_size > 0 && test_size < 1)
    var data = X.zip(y)
    if(shuffle){
      val rng =  new scala.util.Random(seed)
      data = rng.shuffle(data.toList).toArray
    }
    val split_i = y.size - (y.size * test_size).toInt
    val train = data.slice(0,split_i)
    val test = data.slice(split_i,y.length)

    (train.map(_._1),train.map(_._2),test.map(_._1),test.map(_._2))

  }

}
