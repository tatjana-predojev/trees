name := "trees"

version := "0.1"

scalaVersion := "2.12.7"

//val sparkVersion = "3.0.1"
val sparkVersion = "2.4.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "ml.combust.mleap" %% "mleap-spark" % "0.16.0",
  "ml.combust.mleap" %% "mleap-runtime" % "0.16.0"
)
