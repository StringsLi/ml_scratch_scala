package com.strings.model.classification
import breeze.linalg.{DenseMatrix, DenseVector}
import com.strings.data.Data
import com.strings.model.metric.Metric
import org.slf4j.LoggerFactory

class SVMClassification(C:Double = 1.0,
                        tol:Double = 1e-6,
                        max_iter:Int = 100,
                        kernel:(Array[Array[Double]], Array[Double]) => Array[Double]) extends ClassificationModel {
  private val logger = LoggerFactory.getLogger(classOf[SVMClassification])
  var alpha:Array[Double] = _
  var b:Double = _
  var K:Array[Array[Double]] = _
  var sv_inx:Array[Int] = _
  var X:Array[Array[Double]] = _
  var y:Array[Double] = _

  def fit(XX:Array[Array[Double]], yy:Array[Double]):Unit = {
    X = XX
    y = yy
    val n_samples = X.length
    alpha = Array.fill(n_samples)(0.0)
    K = Array.fill(n_samples,n_samples)(0.0)
    sv_inx = Array.range(0,n_samples)
    for(i <- 0 until n_samples){
      K(i) = kernel(X,X(i))  // 有些问题,后面把K转置就OK了
    }
    transpose(K)
    var flag = true
    var iter = 0
    for(_ <- 0 until max_iter if flag) {
      iter += 1
      val alpha_prev = alpha.clone()
      for (j <- 0 until n_samples) {
        val i = selectJ(j,n_samples)
        val e_i = _error(i)
        val e_j = _error(j)
        if ((e_i * y(i) < -tol && alpha(i) < C) || (e_i * y(i) > tol && alpha(i) > 0)) {
          val eta = 2.0 * K(i)(j) - K(i)(i) - K(j)(j)
          if(eta>= 0){
            logger.info("eta >=0 continue")
          }else{
            // 计算L, 和H ,参见相关公式, 主要分i=j和i!=j两种情况
            val L = _find_bounds(i,j)._1
            val H = _find_bounds(i,j)._2
            val alpha_io = alpha(i)
            val alpha_jo = alpha(j)
            alpha(j) -=  y(j) * (e_i - e_j) / eta
            alpha(j) = clipAlpha(alpha(j), H, L)
            alpha(i) = alpha(i) + y(i) * y(j) * (alpha_jo - alpha(j))

            val b1 = b - e_i - y(i) *(alpha(i) - alpha_io)*K(i)(i)
                            -y(j) *(alpha(j) - alpha_jo)*K(i)(j)
            val b2 = (b - e_j - y(j) *(alpha(j) - alpha_jo)*K(j)(j)
                        -y(i)*(alpha(i) - alpha_io)*K(i)(j))

            if (alpha(i) > 0 && alpha(i) < C) b = b1
            else if (alpha(j) > 0 && alpha(j) < C) b = b2
            else b = (b1 + b2) / 2.0
          }
        }
        val diff =  alpha.zip(alpha_prev).map(x => math.abs(x._1 - x._2)).sum
        if(diff < tol){
          flag = false
        }
      }
      logger.info(s"Convergence has reached after $iter.")
    }
    sv_inx = alpha.indices.filter(i => alpha(i) > 0).toArray // 挑选支持向量的编号

  }

  def _find_bounds(i:Int,j:Int):(Double,Double)={
    var L,H = 0.0
    if (y(i) != y(j)) {
      L = math.max(0.0,alpha(j) - alpha(i))
      H = math.min(C,C - alpha(i) - alpha(j))
    }
    else {
      L = math.max(0.0,alpha(j) + alpha(i) - C)
      H = math.min(C,alpha(i) + alpha(j))
    }
    (L,H)
  }

  def transpose(matrix: Array[Array[Double]]): Unit = {
    for (i <- matrix.indices; j <- i + 1 until matrix(0).length) {
      val num = matrix(i)(j)
      matrix(i)(j) = matrix(j)(i)
      matrix(j)(i) = num
    }
  }


  //随机选择第二个变量
  def selectJ(i: Int, m: Int): Int = {
    var j = i
    while (j == i) {
      j = util.Random.nextInt(m)
    }
    j
  }

  def _predict_row(x:Array[Double]):Double = {
    val selectX = sv_inx.map(X(_))
    val k_v = kernel(selectX,x)

    val selectAlpha = sv_inx.map(alpha(_))
    val selectY = sv_inx.map(y(_))

    val part1 = selectAlpha.zip(selectY).map(i => i._2 * i._1)
    val tmp = part1.zip(k_v).map(i => i._1 * i._2).sum
    tmp + b

  }

  def _error(i:Int):Double = {
    _predict_row(X(i)) - y(i)
  }

  def clipAlpha(aj: Double, H: Double, L: Double): Double = if (aj > H) H else if (aj < L) L else aj

  def predict(x: Array[Array[Double]]):Array[Double] = {
    val n = x.length
    val result = Array.fill(n)(0.0)
    for(i <- 0 until n){
      result(i) = math.signum(_predict_row(x(i)))
    }
    result
  }

  override def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    DenseVector.zeros(1)
  }
}

object SVMClassification{
  def main(args: Array[String]): Unit = {

    val data = Data.iris4BinaryClassification()
    val y_trans = data._2.map(x => if(x == 0.0) -1.0 else 1.0)
    val train_test_data = Data.train_test_split(data._1,y_trans,0.3,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    def linear_kernel(x:Array[Array[Double]],y:Array[Double]): Array[Double] ={
      x.map(i => i.zip(y).map(j=>j._1 * j._2).sum)

    }
    val svm = new SVMClassification(kernel = linear_kernel)
    svm.fit(trainX,trainY)

    val pred = svm.predict(testX)
    println(pred.toList)
    val acc =  Metric.accuracy(pred,testY) * 100
    println(f"准确率为: $acc%-5.2f%%")

  }

}
