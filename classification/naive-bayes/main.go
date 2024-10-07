package main

import (
	"fmt"
	"log"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/filters"
	"github.com/sjwhitworth/golearn/naive"
)

// main is the entry point of the program. It performs the following tasks:
// 1. Loads the iris dataset into golearn "instances" from a CSV file.
// 2. Seeds the random number generator to ensure reproducibility.
// 3. Creates a random forest classifier with 10 trees and 2 features per tree.
// 4. Uses cross-fold validation to train and evaluate the model on 5 folds of the dataset.
// 5. Calculates the mean, variance, and standard deviation of the accuracy from the cross-validation results.
// 6. Prints the cross-validation accuracy metrics.
func main() {
	train()
}

// convertToBinary utilizes built in golearn functionality to
// convert our labels to a binary label format.
func convertToBinary(src base.FixedDataGrid) base.FixedDataGrid {
	b := filters.NewBinaryConvertFilter()
	attrs := base.NonClassAttributes(src)
	for _, a := range attrs {
		b.AddAttribute(a)
	}
	b.Train()
	ret := base.NewLazilyFilteredInstances(src, b)
	return ret
}

func train() {
	// Load the loan training dataset into golearn "instances".
	trainingData, err := base.ParseCSVToInstances("../dataset/training.csv", true)
	if err != nil {
		log.Fatal(err)
	}
	// Create a new Naive Bayes classifier.
	nb := naive.NewBernoulliNBClassifier()
	// Train the Naive Bayes classifier.
	nb.Fit(convertToBinary(trainingData))
	// Load the loan test dataset into golearn "instances".
	// Use the training data as a template to ensure the test data format matches.
	testData, err := base.ParseCSVToTemplatedInstances("../dataset/test.csv", true, trainingData)
	if err != nil {
		log.Fatal(err)
	}
	// Make predictions on the test data.
	predictions, err := nb.Predict(convertToBinary(testData))
	if err != nil {
		log.Fatal(err)
	}
	// Generate a confusion matrix.
	cm, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		log.Fatal(err)
	}
	// Calculate and print the accuracy.
	accuracy := evaluation.GetAccuracy(cm)
	fmt.Printf("\nAccuracy: %0.2f\n\n", accuracy)
}
