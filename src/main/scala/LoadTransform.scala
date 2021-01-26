import ml.combust.bundle.BundleFile
import ml.combust.bundle.dsl.Bundle
import ml.combust.mleap.runtime.MleapSupport.MleapBundleFileOps
import ml.combust.mleap.runtime.frame.Transformer
import resource.managed
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import ml.combust.mleap.core.types._

object LoadTransform extends App {

  // load the spark pipeline we saved after training
  val destPipelineDir = "/home/tatjana/work/ml/spark/trees/src/main/resources"
  val bundle: Bundle[Transformer] = (for (bundleFile <- managed(BundleFile(s"jar:file:$destPipelineDir/spark-rf-pipeline.zip")))
    yield {
      bundleFile.loadMleapBundle().get
  }).opt.get

  // create a simple LeapFrame to transform
  val schema = StructType(StructField("elevation", ScalarType.Int),
    StructField("aspect", ScalarType.Int),
    StructField("slope", ScalarType.Int),
    StructField("hdh", ScalarType.Int),
    StructField("vdh", ScalarType.Int),
    StructField("hdr", ScalarType.Int),
    StructField("hillshade_9am", ScalarType.Int),
    StructField("hillshade_noon", ScalarType.Int),
    StructField("hillshade_3pm", ScalarType.Int),
    StructField("hdfp", ScalarType.Int),
    StructField("wilderness", ScalarType.Double),
    StructField("soil", ScalarType.Double)).get
  val data = Seq(Row(1859,18,12,67,11,90,211,215,139,792,3.0,1.0),
    Row(1874,18,14,0,0,90,208,209,135,793,3.0,4.0))
  val frame = DefaultLeapFrame(schema, data)

  // transform the dataframe using our pipeline
  val mleapPipeline = bundle.root
  val frame2 = mleapPipeline.transform(frame).get
  val data2 = frame2.dataset

  // cover_type prediction classes 1 to 7
  println(s"1st example predicted cover type ${data2(0).getDouble(16)}")
  println(s"1st example cover type probabilities ${data2(0).getTensor(15).toArray.mkString("(", ", ", ")")}")
  println(s"2nd example predicted cover type ${data2(1).getDouble(16)}")
  println(s"2nd example cover type probabilities ${data2(1).getTensor(15).toArray.mkString("(", ", ", ")")}")
}
