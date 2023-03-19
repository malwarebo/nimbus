#include "weather_model.h"

WeatherModel::WeatherModel(int num_features) {
  // Create the TensorFlow graph
  Scope root = Scope::NewRootScope();
  input_ = Placeholder(root.WithOpName("input"), DT_FLOAT, Placeholder::Shape({-1, num_features}));
  label_ = Placeholder(root.WithOpName("label"), DT_FLOAT, Placeholder::Shape({-1, 1}));
  hidden_ = Dense(root.WithOpName("hidden"), input_, 10, Dense::Activation::RELU);
  output_ = Dense(root.WithOpName("output"), hidden_, 1);
  loss_ = Mean(root.WithOpName("loss"), SquaredDifference(root, output_, label_), Mean::Reduction::MEAN);
  optimizer_ = ApplyAdam(root.WithOpName("optimizer"), 1e-3, AdamOptimizer::Beta1(0.9), AdamOptimizer::Beta2(0.999), AdamOptimizer::Epsilon(1e-8)).Minimize(loss_);

  // Create the TensorFlow session
  SessionOptions options;
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
  session_ = NewSession(options);
  TF_CHECK_OK(session_->Create(root.ToGraphDef()));
}

void WeatherModel::Train(const vector<vector<float>>& features, const vector<float>& labels, int num_epochs) {
  // Train the TensorFlow model
  Tensor input_tensor(DT_FLOAT, TensorShape({(int)features.size(), (int)features[0].size()}));
  for (int i = 0; i < features.size(); i++) {
    for (int j = 0; j < features[0].size(); j++) {
      input_tensor.matrix<float>()(i, j) = features[i][j];
    }
  }
  Tensor label_tensor(DT_FLOAT, TensorShape({(int)labels.size(), 1}));
  for (int i = 0; i < labels.size(); i++) {
    label_tensor.matrix<float>()(i, 0) = labels[i];
  }
  for (int i = 0; i < num_epochs; i++) {
    TF_CHECK_OK(session_->Run({{input_, input_tensor}, {label_, label_tensor}}, {loss_, optimizer_}, {}));
  }
}

float WeatherModel::Predict(const vector<float>& features) {
  // Use the TensorFlow model to make a prediction
  Tensor input_tensor(DT_FLOAT, TensorShape({1, (int)features.size()}));
  for (int i = 0; i < features.size(); i++) {
    input_tensor.matrix<float>()(0, i) = features[i];
  }
  vector<Tensor> output_tensors;
  TF_CHECK_OK(session_->Run({{input_, input_tensor}}, {output_}, &output_tensors));
  return output_tensors[0].scalar<float>()(0);
}
