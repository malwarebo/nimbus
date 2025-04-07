#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

class DataUtils {
public:
    static void NormalizeFeatures(vector<vector<float>>& features) {
        if (features.empty() || features[0].empty()) return;
        
        size_t feature_count = features[0].size();
        vector<float> min_vals(feature_count, numeric_limits<float>::max());
        vector<float> max_vals(feature_count, numeric_limits<float>::lowest());
        
        for (const auto& sample : features) {
            for (size_t i = 0; i < feature_count; i++) {
                min_vals[i] = min(min_vals[i], sample[i]);
                max_vals[i] = max(max_vals[i], sample[i]);
            }
        }
        
        for (auto& sample : features) {
            for (size_t i = 0; i < feature_count; i++) {
                float range = max_vals[i] - min_vals[i];
                if (range > 0.0001f) { // Avoid division by near-zero
                    sample[i] = (sample[i] - min_vals[i]) / range;
                } else {
                    sample[i] = 0.5f; // Default value if range is too small
                }
            }
        }
    }
    
    // Split data into training and validation sets
    static void SplitData(const vector<vector<float>>& features, const vector<float>& labels,
                   vector<vector<float>>& train_features, vector<float>& train_labels,
                   vector<vector<float>>& val_features, vector<float>& val_labels,
                   float val_ratio = 0.2) {
        vector<size_t> indices(features.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = i;
        }
        
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
        
        size_t val_size = size_t(val_ratio * features.size());
        
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
    
    static float CalculateMAE(const vector<float>& predictions, const vector<float>& actual) {
        if (predictions.size() != actual.size() || predictions.empty()) {
            return -1.0f; // Error
        }
        
        float total_error = 0.0f;
        for (size_t i = 0; i < predictions.size(); i++) {
            total_error += std::abs(predictions[i] - actual[i]);
        }
        
        return total_error / predictions.size();
    }
};

#endif // DATA_UTILS_H
