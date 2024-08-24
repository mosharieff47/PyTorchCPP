#include <iostream>
#include <Python.h>
#include <string>
#include <vector>
#include <map>
#include <math.h>

// This class normalizes and unnormalizes data using the min/max method
class normalize {

    private:
        // All min/max vectors for each variable
        std::vector<double> xminimum, xmaximum, yminimum, ymaximum, tminimum, tmaximum;

        // Transpose matrix function
        std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> x){
            std::vector<std::vector<double>> result;
            std::vector<double> temp;
            for(int i = 0; i < x[0].size(); ++i){
                temp.clear();
                for(int j = 0; j < x.size(); ++j){
                    temp.push_back(x[j][i]);
                }
                result.push_back(temp);
            }
            return result;
        }

        // Prints the shape of the dataset (similar to np.array(x).shape)
        void Shape(std::vector<std::vector<double>> x, std::string name){
            std::cout << name << "\t" << x.size() << "\t" << x[0].size() << std::endl;
        }

    public:
        // Normalizes your dataset using the min/max method
        std::map<std::string, std::vector<std::vector<double>>> Normalize(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, std::vector<std::vector<double>> testing){
            xminimum.clear();
            xmaximum.clear();
            yminimum.clear();
            ymaximum.clear();
            std::map<std::string, std::vector<std::vector<double>>> result;
            std::vector<double> temp, xtemp, ytemp;
            x = transpose(x);
            y = transpose(y);
            testing = transpose(testing);
            for(auto & i : x){
                auto imin = std::min_element(i.begin(), i.end());
                auto imax = std::max_element(i.begin(), i.end());
                xminimum.push_back(static_cast<double>(*imin));
                xmaximum.push_back(static_cast<double>(*imax));
            }
            for(auto & i : y){
                auto imin = std::min_element(i.begin(), i.end());
                auto imax = std::max_element(i.begin(), i.end());
                yminimum.push_back(static_cast<double>(*imin));
                ymaximum.push_back(static_cast<double>(*imax));
            }

            x = transpose(x);
            y = transpose(y);
            testing = transpose(testing);
            for(int i = 0; i < x.size(); ++i){
                xtemp.clear();
                ytemp.clear();
                for(int j = 0; j < x[0].size(); ++j){
                    double num = (x[i][j] - xminimum[j])/(xmaximum[j] - xminimum[j]);
                    xtemp.push_back(num);
                }
                for(int j = 0; j < y[0].size(); ++j){
                    double bum = (y[i][j] - yminimum[j])/(ymaximum[j] - yminimum[j]);
                    ytemp.push_back(bum);
                }
                result["inputs"].push_back(xtemp);
                result["outputs"].push_back(ytemp);
            }
            
            for(int i = 0; i < testing.size(); ++i){
                xtemp.clear();
                for(int j = 0; j < testing[0].size(); ++j){
                    double yahoo = (testing[i][j] - xminimum[j])/(xmaximum[j] - xminimum[j]);
                    xtemp.push_back(yahoo);
                }
                result["testing"].push_back(xtemp);
            }

            return result;
        }

        // This function converts your normalized data back to the data you had initially
        std::vector<std::vector<double>> UnNormalize(std::vector<std::vector<double>> z){
            std::vector<std::vector<double>> result;
            std::vector<double> temp;
            for(int i = 0; i < z.size(); ++i){
                temp.clear();
                for(int j = 0; j < z[0].size(); ++j){
                    double summer = z[i][j]*(ymaximum[j] - yminimum[j]) + yminimum[j];
                    temp.push_back(summer);
                }
                result.push_back(temp);
            }
            return result;
        }

};

class finnet {

    private:
        int N = 0;
        int T = 0;
        int C = 0;
        int yC = 0;
        double learn_rate = 0;
        int epochs = 0;

        // Fetches most current unix timestamp
        double stamp(){
            auto now = std::chrono::system_clock::now();
            std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
            double the_time = (double) now_time_t;
            return the_time;
        }

        // Converts Python code returned as a PyObject into a 2D C++ vector
        std::vector<std::vector<double>> PyToCPP(PyObject* py_list) {
            std::vector<std::vector<double>> result;

            // Check if py_list is actually a list
            if (!PyList_Check(py_list)) {
                PyErr_SetString(PyExc_TypeError, "Expected a Python list");
                return result;
            }

            // Get the length of the outer list (number of rows)
            Py_ssize_t num_rows = PyList_Size(py_list);

            // Iterate through the outer list (rows)
            for (Py_ssize_t i = 0; i < num_rows; ++i) {
                PyObject* inner_list = PyList_GetItem(py_list, i);  // Get each inner list

                // Check if the inner item is also a list
                if (!PyList_Check(inner_list)) {
                    PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
                    result.clear();
                    return result;
                }

                std::vector<double> row;

                // Get the length of the inner list (number of elements in each row)
                Py_ssize_t num_cols = PyList_Size(inner_list);

                // Iterate through the inner list (columns)
                for (Py_ssize_t j = 0; j < num_cols; ++j) {
                    PyObject* item = PyList_GetItem(inner_list, j);  // Get each item

                    // Check if the item is a float or double
                    if (!PyFloat_Check(item) && !PyNumber_Check(item)) {
                        PyErr_SetString(PyExc_TypeError, "Expected a list of lists of doubles");
                        result.clear();
                        return result;
                    }

                    // Convert the item to double and append to row
                    double value = PyFloat_AsDouble(item);
                    row.push_back(value);
                }

                // Add row to result
                result.push_back(row);
            }

            return result;
        }

        // This performes a test from a trained model and returns a 2D vector with the results of the predictions
        std::vector<std::vector<double>> Experiment(PyObject* model, PyObject* a) {
            // Import torch module
            PyObject* torch_module = PyImport_ImportModule("torch");


            // Get torch.no_grad() function
            PyObject* no_grad_func = PyObject_GetAttrString(torch_module, "no_grad");
    

            // Call torch.no_grad() context manager
            PyObject* output = PyObject_CallObject(no_grad_func, nullptr);
            Py_DECREF(no_grad_func);  // Release no_grad_func


            // Get the __enter__() method of the context manager
            PyObject* enter_method = PyObject_GetAttrString(output, "__enter__");
         

            // Call __enter__() to enter the context
            PyObject* enter_result = PyObject_CallObject(enter_method, nullptr);
            Py_DECREF(enter_method);  // Release enter_method


            // Call model(a) to get the output tensor
            PyObject* args = PyTuple_Pack(1, a);  // Pack 'a' into a tuple
         

            PyObject* model_output = PyObject_CallObject(model, args);
            Py_DECREF(args);  // Release args

      

            // Get numpy() from the output tensor
            PyObject* numpy_func = PyObject_GetAttrString(model_output, "numpy");
         

            PyObject* numpy_array = PyObject_CallObject(numpy_func, nullptr);
            Py_DECREF(numpy_func);  // Release numpy_func

   

            // Get tolist() from numpy array
            PyObject* tolist_func = PyObject_GetAttrString(numpy_array, "tolist");
      

            // Call tolist() to get the result list
            PyObject* result = PyObject_CallObject(tolist_func, nullptr);
            Py_DECREF(tolist_func);  // Release tolist_func

            if (!result) {
                PyErr_Print();
            } else {
                //PyObject_Print(result, stdout, Py_PRINT_RAW);  // Print the result list
                //Py_DECREF(result);
            }

            // Clean up
            Py_DECREF(numpy_array);
            Py_DECREF(model_output);
            Py_DECREF(enter_result);
            Py_DECREF(output);
            Py_DECREF(torch_module);

            return PyToCPP(result);
        }

    public:

        // This initializes the neural network class with the proper dimensions
        finnet(int inputs, int input_cols, int ycols, int test_rows, int ep, double lr){
            N = inputs;
            T = test_rows;
            C = input_cols;
            yC = ycols;
            epochs = ep;
            learn_rate = lr;
        }

        // This metric is used to evaluate machine learning models performance
        double RMSE = 0;

        // This generates the neural network model and parameters to be used in training and testing
        std::map<std::string, PyObject*> Model(std::string the_path){
            
            // This sets the path file so your computer can load the neural network Python file
            PyRun_SimpleString("import sys");
            std::string entry_command = "sys.path.append('" + the_path + "')";
            
            PyRun_SimpleString(entry_command.c_str());
        
            PyObject * net = PyImport_Import(PyUnicode_FromString("net"));
            PyObject * modelx = PyObject_GetAttrString(net, "NeuralNet");

            PyObject * modelArgs = PyTuple_New(2);
            PyTuple_SetItem(modelArgs, 0, PyLong_FromLong(C));
            PyTuple_SetItem(modelArgs, 1, PyLong_FromLong(yC));

            // Imports neural network
            PyObject * model = PyObject_CallObject(modelx, modelArgs);

            // This removes the path so you can import other libraries stored in your PATH
            std::string exit_command = "sys.path.remove('" + the_path + "')";
            PyRun_SimpleString(exit_command.c_str());

            PyObject * emptyargs = PyTuple_New(0);

            // Imports PyTorch and its subsets
            PyObject * torch = PyImport_Import(PyUnicode_FromString("torch"));
            PyObject * nn = PyObject_GetAttrString(torch, "nn");
            PyObject * optim = PyObject_GetAttrString(torch, "optim");

            PyObject * mohammed = PyObject_GetAttrString(nn, "MSELoss");
            PyObject * criterion = PyObject_CallObject(mohammed, emptyargs);
            PyObject * Adam = PyObject_GetAttrString(optim, "Adam");

            PyObject * tensorObj = PyObject_GetAttrString(torch, "tensor");
            PyObject * float32 = PyObject_GetAttrString(torch, "float32");
            PyObject * stack = PyObject_GetAttrString(torch, "stack");

            // This vector contains all of the necessary variables to train and test your data
            std::map<std::string, PyObject*> results = {
                {"model", model},
                {"torch", torch},
                {"criterion", criterion},
                {"Adam", Adam},
                {"tensorObj", tensorObj},
                {"float32", float32},
                {"stack", stack}
            };

            return results;
        
        }

        // Responsible for training the model
        std::map<std::string, PyObject*> TrainModel(std::map<std::string, PyObject*> params, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs){
            
            // This loads the number of rows and columns your dataset will be transferred to
            PyObject * XR = PyTuple_New(N);
            PyObject * XC = PyTuple_New(C);
            PyObject * YR = PyTuple_New(N);
            PyObject * YC = PyTuple_New(yC);

            // This loop is run before training to ensure PyObject values are correct
            for(int i = 0; i < N; ++i){
                for(int j = 0; j < C; ++j){
                    PyTuple_SetItem(XC, j, PyFloat_FromDouble(inputs[i][j]));
                }
                for(int j = 0; j < yC; ++j){
                    PyTuple_SetItem(YC, j, PyFloat_FromDouble(outputs[i][j]));
                }
                // This converts generated PyLists and Tuples to convert to PyTorch torch objects for X
                PyObject * torchArgsX = PyTuple_New(1);
                PyObject * torchParamsX = PyDict_New();
                PyTuple_SetItem(torchArgsX, 0, XC);
                PyDict_SetItemString(torchParamsX, "dtype", params["float32"]);
                PyObject * fire = PyObject_Call(params["tensorObj"], torchArgsX, torchParamsX);
                PyTuple_SetItem(XR, i, fire);

                // This converts generated PyLists and Tuples to convert to PyTorch torch objects for Y
                PyObject * torchArgsY = PyTuple_New(1);
                PyObject * torchParamsY = PyDict_New();
                PyTuple_SetItem(torchArgsY, 0, YC);
                PyDict_SetItemString(torchParamsY, "dtype", params["float32"]);
                PyObject * fight = PyObject_Call(params["tensorObj"], torchArgsY, torchParamsY);
                PyTuple_SetItem(YR, i, fight);
            }

            // This performs the torch.stack option which is necessary in training the model
            PyObject * StackX = PyTuple_New(1);
            PyObject * StackY = PyTuple_New(1);

            PyTuple_SetItem(StackX, 0, XR);
            PyTuple_SetItem(StackY, 0, YR);

            PyObject * Xp = PyObject_CallObject(params["stack"], StackX);
            PyObject * Yp = PyObject_CallObject(params["stack"], StackY);

            PyObject * the_params = PyObject_CallMethod(params["model"], "parameters", NULL);

            PyObject * optArgs = PyTuple_New(1);
            PyObject * optPrms = PyDict_New();
            PyTuple_SetItem(optArgs, 0, the_params);
            PyDict_SetItemString(optPrms, "lr", PyFloat_FromDouble(learn_rate));

            // This initializes the optimizer
            PyObject * optimizer = PyObject_Call(params["Adam"], optArgs, optPrms);

            RMSE = 0;

            // Most current time is initialized in order to count seconds elapsed through training
            double T0 = stamp();
            for(int epoch = 1; epoch <= epochs; ++epoch){

                // Passing the X frame to the Neural Network
                PyObject * argModel = PyTuple_New(1);
                PyTuple_SetItem(argModel, 0, Xp);
                PyObject * pred = PyObject_CallObject(params["model"], argModel);

                // Comparing predicted to actual
                PyObject * argCriteria = PyTuple_New(2);
                PyTuple_SetItem(argCriteria, 0, pred);
                PyTuple_SetItem(argCriteria, 1, Yp);

                // Generates the loss between the predicted and actual
                PyObject * loss = PyObject_CallObject(params["criterion"], argCriteria);
                
                // Performs backpropigation steps
                PyObject_CallMethod(optimizer, "zero_grad", NULL);
                PyObject_CallMethod(loss, "backward", NULL);
                PyObject_CallMethod(optimizer, "step", NULL);
                
                // Returns loss in order to compute RMSE
                PyObject * lost = PyObject_CallMethod(loss, "item", NULL);

                double score = PyFloat_AsDouble(lost);
                RMSE += score;

                std::cout << "Epochs Left: " << epochs - epoch << "\tLoss: " << score << std::endl;
                
            }
            double T1 = stamp();

            std::cout << "C++ took " << T1 - T0 << " seconds to train the model" << std::endl;

            RMSE /= ((double) epochs);
            RMSE = pow(RMSE, 0.5);

            return params;
        }

        // Tests the model
        std::vector<std::vector<double>> TestModel(std::map<std::string, PyObject*> params, std::vector<std::vector<double>> test_set){
            std::vector<std::vector<double>> result;
            PyObject * TR = PyTuple_New(T);
            PyObject * TC = PyTuple_New(C);
            
            // All of the testing parameters are converted into PyObjects to work with the neural network
            for(int i = 0; i < T; ++i){
                for(int j = 0; j < C; ++j){
                    PyTuple_SetItem(TC, j, PyFloat_FromDouble(test_set[i][j]));
                    //std::cout << "Test: " << test_set[i][j] << std::endl;
                }
                PyObject * torchArgsZ = PyTuple_New(1);
                PyObject * torchParamsZ = PyDict_New();
                PyTuple_SetItem(torchArgsZ, 0, TC);
                PyDict_SetItemString(torchParamsZ, "dtype", params["float32"]);
                PyObject * fire = PyObject_Call(params["tensorObj"], torchArgsZ, torchParamsZ);
                PyTuple_SetItem(TR, i, fire);
            }

            PyObject * StackZ = PyTuple_New(1);
            PyTuple_SetItem(StackZ, 0, TR);
            PyObject * Tp = PyObject_CallObject(params["stack"], StackZ);

            // Model tests the results are extracts it as a 2D vector
            return Experiment(params["model"], Tp);
        }

};


