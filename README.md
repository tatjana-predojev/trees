## Decision trees

This use case explores decision tree models in spark. Dataset consists of various soil and terrain features such as elevation, slope, hillshade, horizontal and vertical distances to hydrology, roadways etc. We use all these to try to predict the forest cover type which can be one of: Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz (7 different cover types). In short, we have a classification problem with 7 possible classes. Last time I checked, dataset could be downloaded from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/).

## Simple decision tree

First step usually involves some data processing to transform the raw data to the format that spark model to be trained expects. In this case, one hot encoded columns were replaced with categorical column. Very useful helper class here is `VectorAssembler`, a feature transformer that merges multiple columns into a vector column. Another helper class is `VectorIndexer` used for indexing categorical feature columns (i.e. to automatically identify categorical features). Before fitting the model using `DecisionTreeClassifier` we set a pipeline that involves all the described steps (consisting of assembler, indexer and classifier). Finally, `MulticlassClassificationEvaluator` was used to evaluate the performance of the fitted model. In concrete, `accuracy` and `f1 score` were estimated. Both are around `0.7`. 

To get better insights into our fitted model, there are some convenient helper functionalities. One of them prints our entire decision tree, so that we have better idea of how the finall class is reached. 
```
DecisionTreeClassificationModel (uid=dtc_14c8a9c2e8a0) of depth 5 with 45 nodes
  If (feature 0 <= 3041.5)
   If (feature 0 <= 2491.5)
    If (feature 3 <= 15.0)
     If (feature 10 in {2.0})
      Predict: 6.0
     Else (feature 10 not in {2.0})
      If (feature 11 in {0.0,1.0,2.0,3.0,4.0,5.0,10.0,13.0,14.0,16.0,17.0})
       Predict: 4.0
      Else (feature 11 not in {0.0,1.0,2.0,3.0,4.0,5.0,10.0,13.0,14.0,16.0,17.0})
       Predict: 3.0
       ....
```
Another convenient helper function returns feature importances
```
(0.8097408784412716,elevation)
(0.12154964054091946,soil)
(0.021359700526491037,wilderness)
(0.01663557855271323,hdh)
(0.015370705527051469,hdr)
(0.012714539324173386,hillshade_noon)
(0.002628957087379508,hillshade_9am)
(0.0,vdh)
(0.0,slope)
(0.0,hillshade_3pm)
(0.0,hdfp)
(0.0,aspect)
```
We see that `elevation` is by far the most important feature. 

## Random forest

By using a simple decision tree model with default hyperparameters we already get high accuracy. Let us try to improve on that by trying out more powerful random forest model and add validation step to select better hyperparameters. We start by replacing classifier with `RandomForestClassifier`. To define different hyperparameters to validate, we use `ParamGridBuilder`. Besides train and test data, we would need validation data and we select it with `TrainValidationSplit`. Finally, after the model is fit we select the one with best performance. Here are all the hyperparameters
```
  	rfc_f4a70e595514-cacheNodeIds: false,
	rfc_f4a70e595514-checkpointInterval: 10,
	rfc_f4a70e595514-featureSubsetStrategy: auto,
	rfc_f4a70e595514-featuresCol: indexedVector,
	rfc_f4a70e595514-impurity: entropy,
	rfc_f4a70e595514-labelCol: cover_type,
	rfc_f4a70e595514-maxBins: 200,
	rfc_f4a70e595514-maxDepth: 20,
	rfc_f4a70e595514-maxMemoryInMB: 256,
	rfc_f4a70e595514-minInfoGain: 0.0,
	rfc_f4a70e595514-minInstancesPerNode: 1,
	rfc_f4a70e595514-numTrees: 20,
	rfc_f4a70e595514-predictionCol: prediction,
	rfc_f4a70e595514-probabilityCol: probability,
	rfc_f4a70e595514-rawPredictionCol: rawPrediction,
	rfc_f4a70e595514-seed: 207336481,
	rfc_f4a70e595514-subsamplingRate: 1.0
```
With these settings we increase our validation accuracy to `0.95` and test accuracy `0.95`. Significant improvement. 

## Deploying to production

Spark model provides `transform` function to calculate predictions from input data. However, the latency involved and the cost of initiating spark session may be too expensive for production code. One way to solve this problem may be [mleap](https://github.com/combust/mleap). MLeap is an execution engine and serialization format whose goal is to help deploy spark models to production. MLeap runtime can be used to execute our pipeline and algorithm without dependenices on spark context, as can be seen in [`LoadTransform`](src/main/scala/LoadTransform.scala). Remember to serialize models into mleap bundle after fitting. A simple example for our forest cover type gives these results
```
1st example predicted cover type 3.0
1st example cover type probabilities (0.0, 0.0, 0.0038461538461538464, 0.9919051878354204, 0.0, 0.0, 0.00424865831842576, 0.0)
2nd example predicted cover type 6.0
2nd example cover type probabilities (0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.8, 0.0)
```

## Improvements and future work

Would be nice to expose the `predict` function as a web service, using either [http4s](https://http4s.org/) or [akka-http](https://doc.akka.io/docs/akka-http/current/index.html). 
