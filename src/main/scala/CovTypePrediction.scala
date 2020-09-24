import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.spark.SparkSupport.SparkTransformerOps
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.{Model, Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import resource.managed

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
    val wildernessCols = (0 until 4).map(i => s"wilderness_area_$i").toArray
    val train1 = unencodeOneHot(train, wildernessCols, "wilderness")
    val soilCols = (0 until 40).map(i => s"soil_type_$i").toArray
    val train2 = unencodeOneHot(train1, soilCols, "soil")
    //train2.show()
    val test1 = unencodeOneHot(test, wildernessCols, "wilderness")
    val test2 = unencodeOneHot(test1, soilCols, "soil")
    test2.show(10, truncate = false)
    test2.printSchema()

    val pipelineDt = simpleDecisionTree(train2, test2)
    pipelineDt.write.overwrite().save("src/main/resources/pipeline-dt")
    val destPipelineDir = "/home/tatjana/work/ml/spark/trees/src/main/resources"

    serializePipeline(pipelineDt, train2, destPipelineDir, "simple-spark-dt-pipeline.zip")

    val pipelineRf = validatedRandomForest(train2, test2)
    pipelineRf.write.overwrite().save("src/main/resources/pipeline-rf")
    serializePipeline(pipelineRf, train2, destPipelineDir, "spark-rf-pipeline.zip")

  }

  def readData(spark: SparkSession): DataFrame = {
    val dataNoHeader: DataFrame = spark.read
      .option("inferSchema", true)
      .option("header", false)
      .csv("../../datasets/covtype/covtype.data")

    // add header
    val colNames = Seq(
      "elevation", "aspect", "slope",
      "hdh", "vdh", "hdr",
      "hillshade_9am", "hillshade_noon", "hillshade_3pm",
      "hdfp"
    ) ++ (
      (0 until 4).map(i => s"wilderness_area_$i")
      ) ++ (
      (0 until 40).map(i => s"soil_type_$i")
      ) ++ Seq("cover_type")

    dataNoHeader.toDF(colNames:_*).
      withColumn("cover_type", col("cover_type").cast("double"))
  }

  def simpleDecisionTree(train: DataFrame, test: DataFrame): PipelineModel = {

    // data into format that classifier accepts
    val inputCols = train.columns.filter(_ != "cover_type")
    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    // add metadata about categorical columns
    val indexer = new VectorIndexer()
      .setMaxCategories(40) // because soil feature has 40 categories
      .setInputCol("featureVector")
      .setOutputCol("indexedVector")

    val classifier = new DecisionTreeClassifier()
      .setLabelCol("cover_type")
      .setFeaturesCol("indexedVector")
      .setPredictionCol("prediction")
      .setMaxBins(60)

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val model = pipeline.fit(train)

    val predictions = model.transform(test)
    predictions.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("cover_type")
      .setPredictionCol("prediction")
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println("simple decision tree")
    println(s"accuracy = $accuracy")
    println(s"f1 score = $f1")

    model // return the fitted pipeline
  }

  def validatedRandomForest(train: DataFrame, test: DataFrame): PipelineModel = {

    val inputCols = train.columns.filter(_ != "cover_type")
    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    val indexer = new VectorIndexer()
      .setMaxCategories(40) // because soil has 40 categories
      .setInputCol("featureVector")
      .setOutputCol("indexedVector")

    val classifier = new RandomForestClassifier()
      .setLabelCol("cover_type")
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
      .setLabelCol("cover_type")
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

  def serializePipeline(pipeline: PipelineModel, data: DataFrame,
                        destDir: String, pipelineName: String): Unit = {
    val sbc = SparkBundleContext().withDataset(pipeline.transform(data))
    for (bf <- managed(BundleFile(s"jar:file:$destDir/$pipelineName"))) {
      pipeline.writeBundle.format(SerializationFormat.Json).save(bf)(sbc).get
    }
  }

}
