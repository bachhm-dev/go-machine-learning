package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-gota/gota/dataframe"
	"github.com/gonum/matrix/mat64"
	"golang.org/x/exp/rand"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// 1. Remove non-numerical characters from the interest rate and FICO score
// columns.
// 2. Encode our interest rate into two classes for a given interest rate threshold. We
// will use 1.0 to represent our first class (yes, we can get the loan with that
// interest rate) and 0.0 to represent our second class (no, we cannot get the loan
// with that interest rate).
// 3. Select a single value for the FICO credit score. We are given a range of credit
// scores, but we need a single value. The average, minimum, or maximum score
// are natural choices and, in our example, we will use the minimum value (to be
// conservative).
// 4. In this case, we are going to standardize our FICO scores (by subtracting the
// minimum score value from each score and then dividing by the score range).
// This will spread out our score values between 0.0 and 1.0. We need to have a
// justification for this as it is making our data less readable. However, there is a
// good justification. We will be training our logistic regression using a gradient
// descent method that can perform better with normalized data. In fact, when
// running the same example with non-normalized data, there are convergence
// issues

const (
	scoreMax = 830.0
	scoreMin = 640.0
)

func main() {
	dataProfiling()
	savePlotPng()
	splitData()
	train()
	test()
}

func dataProfiling() {
	// Open the loan dataset file.
	f, err := os.Open("../dataset/loan_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 2
	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	// Create the output file.
	f, err = os.Create("../dataset/clean_loan_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a CSV writer.
	w := csv.NewWriter(f)
	// Sequentially move the rows writing out the parsed values.
	for idx, record := range rawCSVData {
		// Skip the header row.
		if idx == 0 {
			// Write the header to the output file.
			if err := w.Write(record); err != nil {
				log.Fatal(err)
			}
			continue
		}
		// Initialize a slice to hold our parsed values.
		outRecord := make([]string, 2)
		// Parse and standardize the FICO score.
		score, err := strconv.ParseFloat(strings.Split(record[0], "-")[0], 64)
		if err != nil {
			log.Fatal(err)
		}
		outRecord[0] = strconv.FormatFloat((score-scoreMin)/(scoreMax-scoreMin), 'f', 4, 64)
		// Parse the Interest rate class.
		rate, err := strconv.ParseFloat(strings.TrimSuffix(record[1], "%"), 64)
		if err != nil {
			log.Fatal(err)
		}
		if rate <= 12.0 {
			outRecord[1] = "1.0"
			// Write the record to the output file.
			if err := w.Write(outRecord); err != nil {
				log.Fatal(err)
			}
			continue
		}
		outRecord[1] = "0.0"
		// Write the record to the output file.
		if err := w.Write(outRecord); err != nil {
			log.Fatal(err)
		}
	}
	// Write any buffered data to the underlying writer (standard output).
	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatal(err)
	}
}

func savePlotPng() {
	// Open the CSV file.
	loanDataFile, err := os.Open("../dataset/clean_loan_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer loanDataFile.Close()
	// Create a dataframe from the CSV file.
	loanDF := dataframe.ReadCSV(loanDataFile)
	// Use the Describe method to calculate summary statistics
	// for all of the columns in one shot.
	loanSummary := loanDF.Describe()
	// Output the summary statistics to stdout.
	fmt.Println(loanSummary) // Create a histogram for each of the columns in the dataset.
	for _, colName := range loanDF.Names() {
		// Create a plotter.Values value and fill it with the
		// values from the respective column of the dataframe.
		plotVals := make(plotter.Values, loanDF.Nrow())
		for i, floatVal := range loanDF.Col(colName).Float() {
			plotVals[i] = floatVal
		}
		// Make a plot and set its title.
		p := plot.New()
		p.Title.Text = fmt.Sprintf("Histogram of a %s", colName)
		// Create a histogram of our values.
		h, err := plotter.NewHist(plotVals, 16)
		if err != nil {
			log.Fatal(err)
		}
		// Normalize the histogram.
		h.Normalize(1)
		// Add the histogram to the plot.
		p.Add(h)
		// Save the plot to a PNG file.
		if err := p.Save(4*vg.Inch, 4*vg.Inch, colName+"_hist.png"); err != nil {
			log.Fatal(err)
		}
	}
}

func splitData() {
	// Open the clean loan dataset file.
	f, err := os.Open("../dataset/clean_loan_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a dataframe from the CSV file.
	// The types of the columns will be inferred.
	loanDF := dataframe.ReadCSV(f)
	// Calculate the number of elements in each set.
	trainingNum := (4 * loanDF.Nrow()) / 5
	testNum := loanDF.Nrow() / 5
	if trainingNum+testNum < loanDF.Nrow() {
		trainingNum++
	}
	// Create the subset indices.
	trainingIdx := make([]int, trainingNum)
	testIdx := make([]int, testNum)
	// Enumerate the training indices.
	for i := 0; i < trainingNum; i++ {
		trainingIdx[i] = i
	}
	// Enumerate the test indices.
	for i := 0; i < testNum; i++ {
		testIdx[i] = trainingNum + i
	}
	// Create the subset dataframes.
	trainingDF := loanDF.Subset(trainingIdx)
	testDF := loanDF.Subset(testIdx)
	// Create a map that will be used in writing the data
	// to files.
	setMap := map[int]dataframe.DataFrame{
		0: trainingDF,
		1: testDF,
	}
	// Create the respective files.
	for idx, setName := range []string{"../dataset/training.csv", "test.csv"} {
		// Save the filtered dataset file.
		f, err := os.Create(setName)
		if err != nil {
			log.Fatal(err)
		}
		// Create a buffered writer.
		w := bufio.NewWriter(f)
		// Write the dataframe out as a CSV.
		if err := setMap[idx].WriteCSV(w); err != nil {
			log.Fatal(err)
		}
	}
}

func train() {
	// Open the training dataset file.
	f, err := os.Open("../dataset/training.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 2
	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	// featureData and labels will hold all the float values that
	// will eventually be used in our training.
	featureData := make([]float64, 2*len(rawCSVData))
	labels := make([]float64, len(rawCSVData))
	// featureIndex will track the current index of the features
	// matrix values.
	var featureIndex int
	// Sequentially move the rows into the slices of floats.
	for idx, record := range rawCSVData {
		// Skip the header row.
		if idx == 0 {
			continue
		}
		// Add the FICO score feature.
		featureVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}
		featureData[featureIndex] = featureVal
		// Add an intercept.
		featureData[featureIndex+1] = 1.0
		// Increment our feature row.
		featureIndex += 2
		// Add the class label.
		labelVal, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Fatal(err)
		}
		labels[idx] = labelVal
	}
	// Form a matrix from the features.
	features := mat64.NewDense(len(rawCSVData), 2, featureData)
	// Train the logistic regression model.
	weights := logisticRegression(features, labels, 100, 0.3) // Output the Logistic Regression model formula to stdout.
	formula := "p = 1 / ( 1 + exp(- m1 * FICO.score - m2) )"
	fmt.Printf("\n%s\n\nm1 = %0.2f\nm2 = %0.2f\n\n", formula, weights[0], weights[1])
}

// logistic implements the logistic function, which
// is used in logistic regression.
func logistic(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// logisticRegression fits a logistic regression model
// for the given data.
func logisticRegression(features *mat64.Dense, labels []float64, numSteps int, learningRate float64) []float64 {
	// Initialize random weights.
	_, numWeights := features.Dims()
	weights := make([]float64, numWeights)
	s := rand.NewSource(uint64(time.Now().UnixNano()))
	r := rand.New(s)
	for idx, _ := range weights {
		weights[idx] = r.Float64()
	}
	// Iteratively optimize the weights.
	for i := 0; i < numSteps; i++ {
		// Initialize a variable to accumulate error for this iteration.
		var sumError float64
		// Make predictions for each label and accumulate error.
		for idx, label := range labels {
			// Get the features corresponding to this label.
			featureRow := mat64.Row(nil, idx, features)
			// Calculate the error for this iteration's weights.
			pred := logistic(featureRow[0] * weights[0] * featureRow[1] * weights[1])
			predError := label - pred
			sumError += math.Pow(predError, 2)
			// Update the feature weights.
			for j := 0; j < len(featureRow); j++ {
				weights[j] += learningRate * predError * pred * (1 - pred) * featureRow[j]
			}
		}
	}
	return weights

}

// predict makes a prediction based on our
// trained logistic regression model.
func predict(score float64) float64 {
	// Calculate the predicted probability.
	p := 1 / (1 + math.Exp(-13.65*score+4.89))
	// Output the corresponding class.
	if p >= 0.5 {
		return 1.0
	}
	return 0.0
}

func test() {
	// Open the test examples.
	f, err := os.Open("../dataset/test.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f) // observed and predicted will hold the parsed observed and predicted values
	// form the labeled data file.
	var observed []float64
	var predicted []float64
	// line will track row numbers for logging.
	line := 1
	// Read in the records looking for unexpected types in the columns.
	for {
		// Read in a row. Check if we are at the end of the file.
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		// Skip the header.
		if line == 1 {
			line++
			continue
		}
		// Read in the observed value.
		observedVal, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Printf("Parsing line %d failed, unexpected type\n", line)
			continue
		}
		// Make the corresponding prediction.
		score, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Printf("Parsing line %d failed, unexpected type\n", line)
			continue
		}
		predictedVal := predict(score)
		// Append the record to our slice, if it has the expected type.
		observed = append(observed, observedVal)
		predicted = append(predicted, predictedVal)
		line++
	}
	// This variable will hold our count of true positive and
	// true negative values.
	var truePosNeg int
	// Accumulate the true positive/negative count.
	for idx, oVal := range observed {
		if oVal == predicted[idx] {
			truePosNeg++
		}
	}
	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(len(observed))
	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
