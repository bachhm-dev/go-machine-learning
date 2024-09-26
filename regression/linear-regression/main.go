package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	"github.com/sajari/regression"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Step 1: Profiling the data
// To make sure that we create a model, or at least process, that we understand, and to
// make sure that we can mentally check our results, we need to start every machine
// learning model building process with data profiling. We need to gain an
// understanding of how each of our variables are distributed and their range and variability.

// Step 2: Choosing our independent variable
// So, now we have some intuition about our data and have come to terms with how our
// data fits within the assumptions of the linear regression model. Now, how do we
// choose which variable to use as our independent variable in trying to predict our
// dependent variable, and average points per game?

// Step 3: Creating our training and test sets
// To avoid over-fitting and make sure that our model can generalize, we are going to
// split our dataset into a training set and a test set
// In this case, we will use an 80/20 split for our training and test data

// Step 4: Training our model
// Next, we are going to actually train, or fit, our linear regression model.
// This just means that we are finding the slope (m) and intercept (b) for the
// line that minimizes the sum of the squared errors

// Step 5: Evaluating the trained model
// We now need to measure the performance of our model to see if we really have any
// power to predict Sales using TV as in independent variable. To do this, we can load
// in our test set, make predictions using our trained model for each test example, and
// then calculate one of the evaluation metrics. In this case, we will use the mean
// absolute error (MAE) to evaluate our model.

const dataset = "../dataset/Advertising.csv"
const trainingDataSet = "../dataset/training.csv"
const testDataSet = "../dataset/test.csv"

func main() {
	dataProfiling()
	chooseIndependentVariable()
	splitData()
	r := train()
	test(r)
	visualizeRegression(r)
}

func dataProfiling() {
	// Open the CSV file.
	advertFile, err := os.Open(dataset)
	if err != nil {
		log.Fatal(err)
	}
	defer advertFile.Close()
	// Create a dataframe from the CSV file.
	advertDF := dataframe.ReadCSV(advertFile)
	// Use the Describe method to calculate summary statistics
	// for all of the columns in one shot.
	advertSummary := advertDF.Describe()
	// Output the summary statistics to stdout.
	fmt.Println(advertSummary)

	// Create a histogram for each of the columns in the dataset.
	for _, colName := range advertDF.Names() {
		// Create a plotter.Values value and fill it with the
		// values from the respective column of the dataframe.
		plotVals := make(plotter.Values, advertDF.Nrow())
		for i, floatVal := range advertDF.Col(colName).Float() {
			plotVals[i] = floatVal
		}
		// Make a plot and set its title.
		p := plot.New()
		p.Title.Text = fmt.Sprintf("Histogram of a %s", colName)
		// Create a histogram of our values drawn
		// from the standard normal.
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

func chooseIndependentVariable() {
	// Open the advertising dataset file.
	f, err := os.Open(dataset)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a dataframe from the CSV file.
	advertDF := dataframe.ReadCSV(f)
	// Extract the target column.
	yVals := advertDF.Col("Sales").Float()
	// Create a scatter plot for each of the features in the dataset.
	for _, colName := range advertDF.Names() {
		// pts will hold the values for plotting
		pts := make(plotter.XYs, advertDF.Nrow())
		// Fill pts with data.
		for i, floatVal := range advertDF.Col(colName).Float() {
			pts[i].X = floatVal
			pts[i].Y = yVals[i]
		}
		// Create the plot.
		p := plot.New()
		p.X.Label.Text = colName
		p.Y.Label.Text = "y"
		p.Add(plotter.NewGrid())
		s, err := plotter.NewScatter(pts)
		if err != nil {
			log.Fatal(err)
		}
		s.GlyphStyle.Radius = vg.Points(3)
		// Save the plot to a PNG file.
		p.Add(s)
		if err := p.Save(4*vg.Inch, 4*vg.Inch, colName+"_scatter.png"); err != nil {
			log.Fatal(err)
		}
	}
}

func splitData() {
	// Open the advertising dataset file.
	f, err := os.Open(dataset)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a dataframe from the CSV file.
	// The types of the columns will be inferred.
	advertDF := dataframe.ReadCSV(f)
	// Calculate the number of elements in each set.
	trainingNum := (4 * advertDF.Nrow()) / 5
	testNum := advertDF.Nrow() / 5
	if trainingNum+testNum < advertDF.Nrow() {
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
	trainingDF := advertDF.Subset(trainingIdx)
	testDF := advertDF.Subset(testIdx)
	// Create a map that will be used in writing the data
	// to files.
	setMap := map[int]dataframe.DataFrame{
		0: trainingDF,
		1: testDF,
	}
	// Create the respective files.
	for idx, setName := range []string{trainingDataSet, testDataSet} {
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

func train() regression.Regression {
	// Open the training dataset file.
	f, err := os.Open(trainingDataSet)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	// Read in all of the CSV records
	reader.FieldsPerRecord = 4
	trainingData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	// In this case we are going to try and model our Sales (y)
	// by the TV feature plus an intercept. As such, let's create
	// the struct needed to train a model using github.com/sajari/regression.
	var r regression.Regression
	r.SetObserved("Sales")
	r.SetVar(0, "TV")
	// Loop of records in the CSV, adding the training data to the regression value.
	for i, record := range trainingData {
		// Skip the header.
		if i == 0 {
			continue
		}
		// Parse the Sales regression measure, or "y".
		yVal, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			log.Fatal(err)
		}
		// Parse the TV value.
		tvVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}
		// Add these points to the regression value.
		r.Train(regression.DataPoint(yVal, []float64{tvVal}))
	}
	// Train/fit the regression model.
	r.Run()
	// Output the trained model parameters.
	fmt.Printf("\nRegression Formula:\n%v\n\n", r.Formula)
	return r
}

func test(r regression.Regression) {
	// Open the test dataset file.
	f, err := os.Open(testDataSet)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	// Read in all of the CSV records
	reader.FieldsPerRecord = 4
	testData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	// Loop over the test data predicting y and evaluating the prediction
	// with the mean absolute error.
	var mAE float64
	for i, record := range testData {
		// Skip the header.
		if i == 0 {
			continue
		}
		// Parse the observed Sales, or "y".
		yObserved, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			log.Fatal(err)
		}
		// Parse the TV value.
		tvVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}
		// Predict y with our trained model.
		yPredicted, err := r.Predict([]float64{tvVal})
		if err != nil {
			log.Fatal(err)
		}
		// Add the to the mean absolute error.
		mAE += math.Abs(yObserved-yPredicted) / float64(len(testData))
	}
	// Output the MAE to standard out.
	fmt.Printf("MAE = %0.2f\n\n", mAE)
}

func visualizeRegression(r regression.Regression) {
	// Output the trained model parameters.
	// Open the advertising dataset file.
	f, err := os.Open(dataset)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// Create a dataframe from the CSV file.
	advertDF := dataframe.ReadCSV(f)
	// Extract the target column.
	yVals := advertDF.Col("Sales").Float()
	// pts will hold the values for plotting.
	pts := make(plotter.XYs, advertDF.Nrow())
	// ptsPred will hold the predicted values for plotting.
	ptsPred := make(plotter.XYs, advertDF.Nrow())
	// Fill pts with data.
	for i, floatVal := range advertDF.Col("TV").Float() {
		pts[i].X = floatVal
		pts[i].Y = yVals[i]
		ptsPred[i].X = floatVal
		ptsPred[i].Y, err = r.Predict([]float64{floatVal})
		if err != nil {
			log.Fatal(err)
		}
	}
	// Create the plot.
	p := plot.New()

	p.X.Label.Text = "TV"
	p.Y.Label.Text = "Sales"
	p.Add(plotter.NewGrid())
	// Add the scatter plot points for the observations.
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	s.GlyphStyle.Radius = vg.Points(3)
	// Add the line plot points for the predictions.
	l, err := plotter.NewLine(ptsPred)
	if err != nil {
		log.Fatal(err)
	}
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	// Save the plot to a PNG file.
	p.Add(s, l)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "regression_line.png"); err != nil {
		log.Fatal(err)
	}
}
