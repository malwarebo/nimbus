#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "models/weather_model.h"

using namespace std;

// Read the historical weather data from a CSV file
bool ReadData(const string& filename, vector<vector<float>>& features, vector<float>& labels) {
  ifstream file(filename);
  
  if (!file.is_open()) {
    cerr << "Error: Unable to open file: " << filename << endl;
    return false;
  }
  
  string line;
  int line_num = 0;
  while (getline(file, line)) {
    line_num++;
    if (line.empty()) continue;
    
    stringstream ss(line);
    string field;
    vector<float> feature;
    float label;
    
    try {
      int i = 0;
      while (getline(ss, field, ',')) {
        if (i == 0) {
          label = stof(field);
        } else {
          feature.push_back(stof(field));
        }
        i++;
      }
      
      // Skip lines with insufficient data
      if (feature.empty()) {
        cerr << "Warning: Skipping line " << line_num << " due to insufficient data" << endl;
        continue;
      }
      
      features.push_back(feature);
      labels.push_back(label);
    } catch (const std::invalid_argument& e) {
      cerr << "Warning: Invalid data at line " << line_num << ", skipping. Error: " << e.what() << endl;
    }
  }
  
  if (features.empty()) {
    cerr << "Error: No valid data found in file: " << filename << endl;
    return false;
  }
  
  return true;
}

// Split data into training and validation sets
void SplitData(const vector<vector<float>>& features, const vector<float>& labels,
               vector<vector<float>>& train_features, vector<float>& train_labels,
               vector<vector<float>>& val_features, vector<float>& val_labels,
               float val_ratio = 0.2) {
  // Create indices
  vector<size_t> indices(features.size());
  for (size_t i = 0; i < indices.size(); i++) {
    indices[i] = i;
  }
  
  // Shuffle indices
  random_shuffle(indices.begin(), indices.end());
  
  // Calculate split point
  size_t val_size = size_t(val_ratio * features.size());
  
  // Split data
  for (size_t i = 0; i < indices.size(); i++) {
    size_t idx = indices[i];
    if (i < val_size) {
      val_features.push_back(features[idx]);
      val_labels.push_back(labels[idx]);
    } else {
      train_features.push_back(features[idx]);
      train_labels.push_back(labels[idx]);
    }
  }
}

int main() {
  try {
    cout << "Weather Prediction System starting up...\n";
    
    // Read the historical weather data
    vector<vector<float>> features;
    vector<float> labels;
    
    string data_file = "weather.csv";
    cout << "Reading data from " << data_file << "..." << endl;
    if (!ReadData(data_file, features, labels)) {
      cerr << "Failed to read data. Exiting." << endl;
      return 1;
    }
    
    cout << "Successfully loaded " << features.size() << " data points." << endl;
    
    // Validate all feature vectors have the same size
    size_t feature_size = features[0].size();
    for (size_t i = 1; i < features.size(); i++) {
      if (features[i].size() != feature_size) {
        cerr << "Error: Inconsistent feature vector size at index " << i << endl;
        return 1;
      }
    }
    
    // Split data into training and validation sets
    vector<vector<float>> train_features, val_features;
    vector<float> train_labels, val_labels;
    
    SplitData(features, labels, train_features, train_labels, val_features, val_labels);
    cout << "Split data: " << train_features.size() << " training samples, " 
         << val_features.size() << " validation samples" << endl;
    
    // Train the weather forecasting model
    cout << "Initializing model..." << endl;
    WeatherModel model(feature_size);
    
    if (!model.GetLastError().empty()) {
      cerr << "Model initialization failed: " << model.GetLastError() << endl;
      return 1;
    }
    
    cout << "Training model..." << endl;
    model.Train(train_features, train_labels, 100);
    
    if (!model.GetLastError().empty()) {
      cerr << "Training failed: " << model.GetLastError() << endl;
      return 1;
    }
    
    // Validate the model
    cout << "Validating model..." << endl;
    float total_error = 0.0f;
    for (size_t i = 0; i < val_features.size(); i++) {
      float prediction = model.Predict(val_features[i]);
      total_error += fabs(prediction - val_labels[i]);
    }
    float avg_error = total_error / val_features.size();
    cout << "Validation MAE: " << avg_error << endl;
    
    // Use the model for a new prediction
    cout << "Making a new prediction..." << endl;
    vector<float> input = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float prediction = model.Predict(input);
    
    if (!model.GetLastError().empty()) {
      cerr << "Prediction failed: " << model.GetLastError() << endl;
      return 1;
    }
    
    cout << "Temperature prediction: " << prediction << endl;
    
    return 0;
  } catch (const std::exception& e) {
    cerr << "Unhandled exception: " << e.what() << endl;
    return 1;
  }
}
