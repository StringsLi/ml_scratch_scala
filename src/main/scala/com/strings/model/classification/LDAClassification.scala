package com.strings.model.classification
import breeze.linalg.{*, DenseMatrix, DenseVector, eig, inv, norm}
import com.strings.utils.Utils
import breeze.stats.mean
import com.strings.data.Data
import com.strings.model.metric.Metric

/**
 * @param k reduce k dimension
 */

class LDAClassification(val k:Int) extends ClassificationModel {

  var centerVectors:Iterable[(Double, DenseVector[Double])] = _
  var weights:DenseMatrix[Double] = _

  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    X(*,::).map(x => predict(x.toArray))
  }

  def fit2(data: List[DenseVector[Double]], labels: List[Double]) = {
    val sample = labels.zip(data)
    computeLDA(sample,k)
  }

  def fit(data: DenseMatrix[Double], labels: List[Double]) = {
      val data_arr:List[DenseVector[Double]] = data(*,::).toIndexedSeq.toList.map(x => x.t)
      fit2(data_arr,labels)
  }

  def computeLDA(dataAndLabels: List[(Double, DenseVector[Double])],k:Int)= {

    val featuresWithlabel = dataAndLabels.groupBy(_._1).values.map(x => (x.map(_._1), Utils.rowsToMatrix(x.map(_._2))))
    val meanVector = featuresWithlabel.map(f => (f._1.toArray.apply(0), mean(f._2(::, *)))) // 对行向量求平均值 each mean is a row vector, not col
    val meanByClass = meanVector.map(_._2)
    val featuresByClass = featuresWithlabel.map(_._2)

    //类内散度矩阵
    val Sw = featuresByClass.zip(meanByClass).map(f => {
      val featuresMinusMean: DenseMatrix[Double] = f._1(*, ::) - f._2.t // row vector, not column
      featuresMinusMean.t * featuresMinusMean: DenseMatrix[Double]
    }).reduce(_+_)

    val numByClass = featuresByClass.map(_.rows : Double)
    val features = Utils.rowsToMatrix(dataAndLabels.map(_._2))
    val totalMean = mean(features(::, *)) // A row-vector, not a column-vector

    val Sb = meanByClass.zip(numByClass).map {
      case (classMean, classNum) => {
        val m = classMean - totalMean
        (m.t * m : DenseMatrix[Double]) :* classNum : DenseMatrix[Double]
      }
    }.reduce(_+_)

    val eigen = eig((inv(Sw): DenseMatrix[Double]) * Sb)
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _).toDenseMatrix.t)
    val topEigenvectors = eigenvectors.zip(eigen.eigenvalues.toArray).sortBy(x => -scala.math.abs(x._2)).map(_._1).take(k)
    weights = DenseMatrix.horzcat(topEigenvectors:_*)
    centerVectors = meanVector.map(f => (f._1,f._2.t))
  }

  def predict(feature:Array[Double]):Double = {
    val numOfClass = centerVectors.map(_._1).toSet.size
    val projection:Array[Double] = Array.fill(numOfClass)(0.0)
    for(i <- 0 until numOfClass){
      val center = centerVectors.filter( f => f._1 == i).map(f => f._2).head
      val l = DenseMatrix(feature)*weights - (DenseMatrix(center)*weights)
      projection(i) = norm(l.toDenseVector,2)
    }
    projection.zipWithIndex.sortBy(_._1).head._2.toDouble
  }

}

object LDAClassification{
  def main(args: Array[String]): Unit = {
    val data = Data.iris4MutilClassification()
    val train_test_data = Data.train_test_split(data._1,data._2,0.2,seed = 1224L)
    val trainX = train_test_data._1
    val trainY = train_test_data._2
    val testX = train_test_data._3
    val testY = train_test_data._4

    val lda = new LDAClassification(3)
    lda.fit(DenseMatrix(trainX:_*),trainY.toList)
    val pred = lda.predict(DenseMatrix(testX:_*))
    val acc =  Metric.accuracy(pred.toArray,testY) * 100
    println(f"准确率为: $acc%-5.2f%%")

  }
}
