#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "models/weather_model.h"

using namespace std;

// Read the historical weather data from a CSV file
void ReadData(const string& filename, vector<vector<float>>& features, vector<float>& labels) {
  ifstream file(filename);
  string line;
  while (getline(file, line)) {
    stringstream ss(line);
    string field;
    vector<float> feature;
    float label;
    int i = 0;
    while (getline(ss, field, ',')) {
      if (i == 0) {
        label = stof(field);
      } else {
        feature.push_back(stof(field));
      }
      i++;
    }
    features.push_back(feature);
    labels.push_back(label);
  }
}

int main() {
  // Read the historical weather data
  vector<vector<float>> features;
  vector<float> labels;
  ReadData("weather.csv", features, labels);

  // Train the weather forecasting model
  WeatherModel model(features[0].size());
  model.Train(features, labels, 100);

  // Use the weather forecasting model to make predictions
  vector<float> input = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  float prediction = model.Predict(input);
  cout << "Temperature prediction: " << prediction << endl;

  return 0;
}
