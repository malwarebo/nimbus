#ifndef WEATHER_MODEL_H
#define WEATHER_MODEL_H

#include <tensorflow/c/c_api.h>
#include <vector>
#include <string>

using std::vector;
using std::string;

class WeatherModel {
public:
  WeatherModel(int num_features);
  ~WeatherModel();
  
  void Train(const vector<vector<float>>& features, const vector<float>& labels, int num_epochs);
  float Predict(const vector<float>& features);
  string GetLastError() const { return last_error_; }

private:
  TF_Session* session_;
  TF_Graph* graph_;
  TF_Status* status_;
  
  int num_features_;
  
  string last_error_;
  
  void CreateSimpleModel();
  void CleanupSession();
};

#endif // WEATHER_MODEL_H
