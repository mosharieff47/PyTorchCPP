#include <iostream>
#include <fstream>
#include <sstream>
#include <Python.h>
#include "finnet.h"
#include "matplotlibcpp.h"
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <curl/curl.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace boost::property_tree;

namespace plt = matplotlibcpp;

std::vector<std::vector<double>> TRANSPOSE(std::vector<std::vector<double>> x)
{
    std::vector<std::vector<double>> res;
    std::vector<double> temp;

    int m = x.size(), n = x[0].size();

    // Shifts the rows and columns to be row=column, column=row
    for(int i = 0; i < n; ++i){
        temp.clear();
        for(int j = 0; j < m; ++j){
            temp.push_back(x[j][i]);
        }
        res.push_back(temp);
    }

    return res;
}

std::vector<double> extract_column(std::vector<std::vector<std::string>> df, int key){
    std::vector<double> result;
    for(int i = 1; i < df.size(); ++i){
        result.push_back(atof(df[i][key].c_str()));
    }
    return result;
}

std::vector<std::vector<std::string>> read_data(std::string filename){
    std::vector<std::vector<std::string>> result;
    std::vector<std::string> temp;
    std::ifstream file(filename);
    std::string row, col;
    while(file >> row){
        temp.clear();
        std::stringstream ss(row);
        while(std::getline(ss, col, ',')){
            temp.push_back(col);
        }
        result.push_back(temp);
    }

    return result;
}

std::vector<std::vector<double>> AddMetrics(std::vector<double> data, int window){
    auto mean = [](std::vector<double> x){
        double total = 0;
        double n = x.size();
        for(auto & i : x){
            total += i;
        }
        total /= n;
        return total;
    };
    
    auto stdev = [&](std::vector<double> x){
        double mu = mean(x);
        double total = 0;
        double n = x.size();
        for(auto & i : x){
            total += pow(i - mu, 2);
        }
        return pow(total/(n - 1), 0.5);
    };
    
    std::vector<std::vector<double>> result;
    for(int i = window; i < data.size(); ++i){
        std::vector<double> hold = {data.begin() + (i - window), data.begin() + i};
        double mean1 = mean(hold);
        double stde1 = stdev(hold);
        result.push_back({data[i], mean1, data[i] - 2.0*stde1, data[i] + 2.0*stde1});
    }

    return result;
}

std::map<std::string, std::vector<std::vector<double>>> Execute(int window, int output, double prop, std::vector<std::vector<double>> DF)
{
    std::map<std::string, std::vector<std::vector<double>>> result;
    int I = int(prop*DF.size());
    std::vector<std::vector<double>> trainX = {DF.begin(), DF.begin() + I}, testX = {DF.end() - window, DF.end()};

    std::vector<std::vector<double>> hold_inputs, hold_outputs;
    std::vector<double> tinputs, toutputs;

    for(int i = window; i < I - output; ++i){
        hold_inputs = {trainX.begin() + (i - window), trainX.begin() + i};
        hold_outputs = {trainX.begin() + i, trainX.begin() + i + output};
        hold_inputs = TRANSPOSE(hold_inputs);
        hold_outputs = TRANSPOSE(hold_outputs);
        tinputs.clear();
        for(auto & f : hold_inputs){
            for(auto & g : f){
                tinputs.push_back(g);
            }
        }
            
        result["inputs"].push_back(tinputs);
        result["outputs"].push_back(hold_outputs[0]);
    }

    testX = TRANSPOSE(testX);
    toutputs.clear();
    for(auto & f : testX){
        for(auto & g : f){
            toutputs.push_back(g);
        }
    }
    result["testing"].push_back(toutputs);

    return result;
    
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
    } catch (std::bad_alloc &e) {
        // Handle memory problem
        return 0;
    }
    return newLength;
}

ptree JSON(std::string message){
    ptree result;
    std::stringstream ss(message);
    read_json(ss, result);
    return result;
}

std::string request(std::string ticker) {
    std::string url = "https://api.exchange.coinbase.com/products/" + ticker + "/candles?granularity=60";
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    return readBuffer;
}

std::vector<double> Parse(std::string ticker){
    std::vector<double> result;
    ptree data = JSON(request(ticker));
    int count = 0;
    for(ptree::const_iterator it = data.begin(); it != data.end(); ++it){
        count = 0;
        for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
            if(count == 4){
                result.push_back(jt->second.get_value<double>());
            }
            count += 1;
        }
    }
    return result;
}

int main()
{
    Py_Initialize();

    normalize norm;

    std::string asset = "BTC-USD";
    
    std::vector<double> close = Parse(asset);
    std::reverse(close.begin(), close.end());
    
    int lookback = 100;
    int window = 100;
    int output = 50;
    int epochs = 1000;
    double learning_rate = 0.00005;
    
    std::vector<std::vector<double>> init_data = AddMetrics(close, lookback);
    std::map<std::string, std::vector<std::vector<double>>> nnet_data, norm_data;
    
    
    nnet_data = Execute(window, output, 0.95, init_data);
    
    
    int rows = nnet_data["inputs"].size();
    int cols = nnet_data["inputs"][0].size();
    int ycols = nnet_data["outputs"][0].size();
    int test_rows = nnet_data["testing"].size();
    int test_cols = nnet_data["testing"][0].size();
    

    std::string address = "/Users/mo/Desktop/PyCPP/neural";

    finnet neural(rows, cols, ycols, test_rows, epochs, learning_rate);

    norm_data = norm.Normalize(nnet_data["inputs"], nnet_data["outputs"], nnet_data["testing"]);


    std::map<std::string, PyObject*> params = neural.Model(address);
    
    
    params = neural.TrainModel(params, norm_data["inputs"], norm_data["outputs"]);
    
    std::vector<std::vector<double>> yp = neural.TestModel(params, norm_data["testing"]);
    yp = TRANSPOSE(yp);

    yp = norm.UnNormalize(yp);
    
    std::cout << "Neural RMSE: " << neural.RMSE << std::endl;
    
    std::vector<double> Xh, Yh, Xg, Yg;
    for(int i = 0; i < test_cols; ++i){
        Xh.push_back(i+1);
        Yh.push_back(nnet_data["testing"][0][i]);
    }

    for(int i = test_cols; i < test_cols + output; ++i){
        Xg.push_back(i-1);
        Yg.push_back(yp[i-test_cols][0]);
    }

    

    
    PyObject * ax = plt::chart2D(111);

    plt::SetChartTitle(ax, asset + " | RMSE = " + std::to_string(neural.RMSE));
    plt::plot2D(ax, Xh, Yh, "red");
    plt::plot2D(ax, Xg, Yg, "limegreen");

    plt::show();
    


    Py_Finalize();

    return 0;
}
