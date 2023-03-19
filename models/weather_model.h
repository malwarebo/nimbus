#ifndef WEATHER_MODEL_H
#define WEATHER_MODEL_H

#include <tensorflow/c/c_api.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/math_ops.h>

using namespace tensorflow;
using namespace tensorflow::ops;

class WeatherModel {
public:
  WeatherModel(int num_features);
  void Train(const vector<vector<float>>& features, const vector<float>& labels, int num_epochs);
  float Predict(const vector<float>& features);

private:
  Session* session_;
  Placeholder input_;
  Placeholder label_;
  Dense hidden_;
  Dense output_;
  Mean loss_;
  Operation optimizer_;
};

#endif // WEATHER_MODEL_H
