# Nimbus - Weather Forecasting with TensorFlow and C++

Nimbus is a weather forecasting application that uses machine learning to predict weather for a given location. It is built using TensorFlow and C++.

## Requirements

    macOS or Linux
    CMake 3.5 or later
    TensorFlow 2.6 or later
    C++11 or later

## Setup

### To set up the project on your system, follow these steps:

Install TensorFlow on your system by following the instructions in the official documentation.

1. Clone the repository:

```bash
git clone https://github.com/<username>/skynet.git
cd skynet
```

2. Create a build directory and configure

```bash
mkdir build
cd build
cmake ..
```

3. To build the project, run the following command from the build directory:

```bash
make
```

## Usage
```bash
./nimbus <path_to_data_file>
```
Replace <path_to_data_file> with the path to a CSV file containing weather data for a specific location. The file should have the following format:
```csv
Date,Temperature (C),Humidity (%),Wind Speed (km/h),Pressure (hPa),Weather
2018-01-01,2.2,93,7,1022,Rain
2018-01-02,1.7,94,8,1021,Rain
2018-01-03,1.5,95,9,1020,Rain
...
```
The application will use the data in this file to train a machine learning model, and then use the model to make predictions for future dates.

## Contributing

Contributions to the project are welcome. 

