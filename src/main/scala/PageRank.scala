import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object PageRank {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    // set up environment

    val conf = new SparkConf()
      .setAppName("PageRank")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    // load ratings and movie titles

    val input = args(0)

    val pages = sc.textFile(new File(input).toString)

    var pageRank = pages.map(x => ((x.split("\\s+"))(0),(x.split("\\s+"))(1).toDouble))
    val pageLinks = pages.map(x => ((x.split("\\s+"))(0), (x.split("\\s+"))(2).replaceAll("[(){}]","").split(",")))

    for (i <- 1 to 100) {
      val contribs = pageLinks.join(pageRank).values.flatMap{ case (urls, rank) =>
        val size = urls.size
        urls.map(url => (url, rank / size))
      }
      pageRank = contribs.filter(x => (x._1.length > 0)).reduceByKey(_ + _)
    }
    pageRank.sortBy(_._2,false).collect().foreach(tup => println(s"${tup._1} has rank:  ${tup._2} ."))

    // clean up
    sc.stop()
  }
}