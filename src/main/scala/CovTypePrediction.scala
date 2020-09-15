import org.apache.spark.ml.{Model, Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object CovTypePrediction {

  def main(args: Array[String]) {

    val spark = SparkSession.builder
      .appName("CovTypePrediction")
      .config("spark.master", "local")
      .getOrCreate()

    val data = readData(spark)

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

    val pipelineDt = simpleDecisionTree(train2, test2)
    pipelineDt.write.overwrite().save("src/main/resources/pipeline-dt")

    val pipelineRf = validatedRandomForest(train2, test2)
    pipelineRf.write.overwrite().save("src/main/resources/pipeline-rf")

  }

  def readData(spark: SparkSession): DataFrame = {
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

    dataNoHeader.toDF(colNames:_*).
      withColumn("Cover_Type", col("Cover_Type").cast("double"))
  }

  def simpleDecisionTree(train: DataFrame, test: DataFrame): PipelineModel = {

    // data into format that classifier accepts
    val inputCols = train.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    // add metadata about categorical columns
    val indexer = new VectorIndexer()
      .setMaxCategories(40) // because soil feature has 40 categories
      .setInputCol("featureVector")
      .setOutputCol("indexedVector")

    val classifier = new DecisionTreeClassifier()
      .setLabelCol("Cover_Type")
      .setFeaturesCol("indexedVector")
      .setPredictionCol("prediction")
      .setMaxBins(60)

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val model = pipeline.fit(train)

    val predictions = model.transform(test)
    //predictions.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println("simple decision tree")
    println(s"accuracy = $accuracy")
    println(s"f1 score = $f1")

    model // return the fitted pipeline
  }

  def validatedRandomForest(train: DataFrame, test: DataFrame): PipelineModel = {

    val inputCols = train.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    val indexer = new VectorIndexer()
      .setMaxCategories(40) // because soil has 40 categories
      .setInputCol("featureVector")
      .setOutputCol("indexedVector")

    val classifier = new RandomForestClassifier()
      .setLabelCol("Cover_Type")
      .setFeaturesCol("indexedVector")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.impurity, Seq("gini", "entropy"))
      .addGrid(classifier.maxDepth, Seq(10, 20))
      .addGrid(classifier.maxBins, Seq(60, 200))
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val validator = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    val validatorModel = validator.fit(train)
    val bestModel = validatorModel.bestModel
    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    println("random forest")
    println(s"validation accuracy ${validatorModel.validationMetrics.max}")
    println(s"test accuracy ${evaluator.evaluate(bestModel.transform(test))}")

    bestModel.asInstanceOf[PipelineModel]
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
