package com.strings.utils

import java.io.{File, PrintWriter}
import breeze.linalg.DenseMatrix

object FileUtils {

  def writeFile(W:DenseMatrix[Double],file:String): Unit ={
    val writer = new PrintWriter(new File(file))
    for(i <- Range(0,W.rows)){
      val row = W(i,::).t.toArray
      writer.write(row.mkString(",") + "\n")
    }
    writer.close()
  }

}
