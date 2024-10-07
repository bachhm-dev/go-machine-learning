package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
)

func main() {
	// Load the iris dataset into golearn "instances".
	irisData, err := base.ParseCSVToInstances("../dataset/iris.csv", true)
	if err != nil {
		log.Fatal(err)
	}
	// Seed the random number generator for reproducibility.
	rand.Seed(44111342)
	// Initialize the ID3 decision tree with a train-prune split parameter of 0.6.
	decisionTree := trees.NewID3DecisionTree(0.6)
	// Perform 5-fold cross-validation to train and evaluate the model.
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, decisionTree, 5)
	if err != nil {
		log.Fatal(err)
	}
	// Calculate the mean, variance, and standard deviation of the accuracy.
	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance)
	// Print the cross-validation accuracy metrics.
	fmt.Printf("\nAccuracy\n%.2f (+/- %.2f)\n\n", mean, stdev*2)
}
