package main

import (
	"fmt"
	"log"
	"math"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"golang.org/x/exp/rand"
)

// main is the entry point of the program. It performs the following tasks:
// 1. Loads the iris dataset into golearn "instances" from a CSV file.
// 2. Seeds the random number generator to ensure reproducibility.
// 3. Creates a random forest classifier with 10 trees and 2 features per tree.
// 4. Uses cross-fold validation to train and evaluate the model on 5 folds of the dataset.
// 5. Calculates the mean, variance, and standard deviation of the accuracy from the cross-validation results.
// 6. Prints the cross-validation accuracy metrics.
func main() {
	// Load the iris dataset into golearn "instances".
	irisData, err := base.ParseCSVToInstances("../dataset/iris.csv", true)
	if err != nil {
		log.Fatal(err)
	}

	rand.Seed(44111342)

	// Create a random forest with 10 trees and 2 features per tree.
	// Typically, the number of features per tree is set to the square root of the total number of features.
	rf := ensemble.NewRandomForest(10, 2)
	// Use cross-fold validation to successively train and evaluate the model
	// on 5 folds of the data set.
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, rf, 5)
	if err != nil {
		log.Fatal(err)
	}
	// Calculate the mean, variance, and standard deviation of the accuracy.
	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance)
	// Print the cross-validation accuracy metrics.
	fmt.Printf("\nAccuracy\n%.2f (+/- %.2f)\n\n", mean, stdev*2)
}
