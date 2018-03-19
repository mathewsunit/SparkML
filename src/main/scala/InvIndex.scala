import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object InvIndex {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    // set up environment

    val conf = new SparkConf()
      .setAppName("InvIndex")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    // load ratings and movie titles

    val trainHomeDir = args(0)
    val words = args(1)

    var docs = sc.emptyRDD[(String,(Int, String))]
    var docCounts = sc.emptyRDD[(String,Int)]

    def getListOfSubDirectories(directoryName: String): Array[String] = {
      (new File(directoryName))
        .listFiles
        .filter(_.isDirectory)
        .map(_.getName)
    }
    def getListOfFiles(directoryName: String): Array[String] = {
      (new File(directoryName))
        .listFiles
        .map(_.getName)
    }
    for (lines <- getListOfSubDirectories(trainHomeDir)) {
      val counts = sc.textFile("file:/"+trainHomeDir+"/"+lines+"/*").flatMap(line => line.split("\\s+"))
        .filter(_.toLowerCase().matches("[a-zA-Z]+")).filter(_.length>5).map(word => (word.toLowerCase(), 1))
        .reduceByKey(_ + _)
      docs = docs.union(counts.map(x => (x._1,(x._2,lines))))
      docCounts = docCounts.union(counts.map(x => (x._1,1)))
    }

    def calculate(tf: Int, df: Int, numDocs: Int): Double = {
      math.sqrt(tf) * (math.log(numDocs / (df).toDouble))
    }
    docCounts = docCounts.reduceByKey((x,y)=>(x+y))

    val tfidf = docs.join(docCounts).map(x => (x._1, x._2._1._2, calculate(x._2._1._1, x._2._2, 20)))

    tfidf.collect().foreach(println)

    sc.stop()
  }
}