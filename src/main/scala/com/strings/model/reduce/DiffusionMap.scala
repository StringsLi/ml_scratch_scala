package com.strings.model.reduce

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, eig, sum}
import breeze.numerics.{exp, pow}
import com.strings.data.Data
import com.strings.utils.{FileUtils, Utils}

/**
 *
 * @param d the dimension.
 * @param sigma the sigma of the gaussian for the gram matrix transformation.
 * @param t he scale of the diffusion (amount of time steps).
 */

class DiffusionMap(var d:Int = 2, sigma:Double = 100, t:Int = 2) {

  def transform(X:List[DenseVector[Double]]):DenseMatrix[Double] = {
    val X_dist = pow(DenseMatrix(Utils.pair_distance(X):_*),2) :/(-sigma.toDouble)
    val K = exp(X_dist)
    val colUSum:DenseVector[Double] = sum(K,Axis._1)

    val KK = K(::,*).map(x => x :/ colUSum)
    val eigen = eig(KK)
    val eigenVectors = eigen.eigenvectors(::,*).toIndexedSeq
    val eigenValues = eigen.eigenvalues.toArray
    val topEigenvectorNvalues = eigenVectors.zip(eigenValues).sortBy(x => -x._2).slice(1,1+d)
    val topValues = DenseVector(topEigenvectorNvalues.map(_._2):_*)
    val vi = DenseMatrix(topEigenvectorNvalues.map(_._1):_*)
    vi.t(*,::).map(x => x :* topValues.map(math.pow(_,t.toDouble)))

  }


}

object DiffusionMap{
  def main(args: Array[String]): Unit = {
    val data = Data.irisData.map(x => DenseVector(x.slice(0,4))).toList
    val diffusionMap = new DiffusionMap()
    val W = diffusionMap.transform(data)
    val file = "D:\\data\\iris_diffmap.txt"
    FileUtils.writeFile(W,file)
  }
}
