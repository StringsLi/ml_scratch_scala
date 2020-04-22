package com.strings.model.reduce

import breeze.linalg.{*, DenseMatrix, DenseVector, eig, inv}
import com.strings.utils.Utils
import breeze.stats.mean
import com.strings.data.Data

/**
 * 线性判别分析用于降维
 */

object LDA {

  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x => DenseVector(x.slice(0,4))).toList
    val label = Data.irisData.map(_.apply(4).toInt).toList
    val ss = fit(data,label,2)
    println(ss)

  }

  def fit(data: List[DenseVector[Double]], labels: List[Int],k:Int) = {
    val sample = labels.zip(data)
    fit2(sample,k)
  }

  def fit2(dataAndLabels: List[(Int, DenseVector[Double])],k:Int)= {

    val featuresByClass = dataAndLabels.groupBy(_._1).values.map(x => Utils.rowsToMatrix(x.map(_._2)))
    val meanByClass = featuresByClass.map(f => mean(f(::, *))) // 对行向量求平均值 each mean is a row vector, not col

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
    val W = DenseMatrix.horzcat(topEigenvectors:_*)
    (W,Sb,Sw)
  }

}
