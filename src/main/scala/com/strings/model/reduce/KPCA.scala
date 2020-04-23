package com.strings.model.reduce

import breeze.linalg.{*, DenseMatrix, DenseVector, Transpose, eig}
import breeze.numerics.{exp, log}
import breeze.stats.mean
import com.strings.data.Data
import com.strings.utils.{FileUtils, Utils}

class KPCA(val kernel_type:String = "rbf") {


  def transform(X:List[DenseVector[Double]],n_components:Int = 3,beta:Int = 10) = {
    val mat:DenseMatrix[Double] = DenseMatrix(Utils.pair_distance(X):_*)
    val K = kernel_type match {
      case "linear" => DenseMatrix(X: _*) * DenseMatrix(X: _*).t
      case "rbf" => exp(mat :* (-beta.toDouble))
      case "log" => log(mat :+ 1.0) :* (-1.0)
    }

    val M_r:Transpose[DenseVector[Double]] = mean(K(::,*))
    val M_c:DenseVector[Double] = mean(K(*,::))
    val meanDist:Double = mean(K)

    // 对核矩阵进行标准化
    val K_std = K(::,*).map(x => x - M_c - M_r.t) :+ meanDist

    val eigen = eig(K_std)
    val eigenvectors = (0 until eigen.eigenvectors.cols).map(eigen.eigenvectors(::, _))
    val eigenValues = eigen.eigenvalues.toArray
    val topEigenvectorsNvalues = eigenvectors.zip(eigenValues).sortBy(x => -x._2).take(n_components)
    val topEigenvectors = topEigenvectorsNvalues.map(_._1)
    val topEigenvalues = topEigenvectorsNvalues.map(_._2)
    val eigVal = DenseVector(topEigenvalues.map(math.sqrt):_*)
    val W = DenseMatrix(topEigenvectors:_*).t
    val vi =W(*,::).map(x => x :/ eigVal)

    K * vi
  }

}

object KPCA{
  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x => DenseVector(x.slice(0,4))).toList

    val kpca = new KPCA()
    val W = kpca.transform(data,n_components = 2)

    val file = "D:\\data\\iris_kpca.txt"

    FileUtils.writeFile(W,file)

  }
}
