import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector

import scala.util.Random

object CovTypePrediction {

  def main(args: Array[String]) {

    val spark = SparkSession.builder
      .appName("CovTypePrediction")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val dataNoHeader: DataFrame = spark.read
      .option("inferSchema", true)
      .option("header", false)
      .csv("../../datasets/covtype/covtype.data")

    // add header
    val colNames = Seq(
      "Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    ) ++ (
      (0 until 4).map(i => s"Wilderness_Area_$i")
      ) ++ (
      (0 until 40).map(i => s"Soil_Type_$i")
      ) ++ Seq("Cover_Type")

    val data = dataNoHeader.toDF(colNames:_*).
      withColumn("Cover_Type", col("Cover_Type").cast("double")) // WHY cast to double??

    val Array(train, test) = data.randomSplit(Array(0.9, 0.1))
    train.cache()
    test.cache()

    // replace one-hot encoded columns with one categorical column
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray
    val train1 = unencodeOneHot(train, wildernessCols, "wilderness")
    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray
    val train2 = unencodeOneHot(train1, soilCols, "soil")
    //train2.show()
    val test1 = unencodeOneHot(test, wildernessCols, "wilderness")
    val test2 = unencodeOneHot(test1, soilCols, "soil")

    val inputCols = train2.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    val indexer = new VectorIndexer()
      .setMaxCategories(40)
      .setInputCol("featureVector")
      .setOutputCol("indexedVector")

    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("indexedVector")
      .setPredictionCol("prediction")
      .setMaxBins(60)

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val model = pipeline.fit(train2)

    val predictions = model.transform(test2)
    predictions.show()
    
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println(s"test error = ${(1.0 - accuracy)}")
    println(s"f1 score = $f1")


  }

  def unencodeOneHot(data: DataFrame,
                     oneHotCols: Array[String],
                     outputCol: String): DataFrame = {

    val assembler = new VectorAssembler()
      .setInputCols(oneHotCols)
      .setOutputCol(outputCol)

    val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

    assembler.transform(data)
      .drop(oneHotCols: _*)
      .withColumn(outputCol, unhotUDF(col(outputCol)))
  }

}
