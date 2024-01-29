# Question 1

In addition to the other graph analysis passes, namely `add_common_metadata_analysis_pass` and `init_metadata_analysis_pass`, the function `report_graph_analysis_pass` is responsible for generating a summary of the Mase graph analysis. It generates a report for the graph analysis and prints out an overview the model in tabular form. Unlike a transform pass, this function does not alter the graph itself; instead, it outputs to a specified file if provided, or prints the summary to the console otherwise. 

The summary includes a count of various node operation types and module types identified within the graph. The graph is composed of sequence blocks describing Mase operators, such as BatchNorm1d, ReLU, Linear, and ReLU, along with their respective parameters (for the JSC_Tiny network).

`placeholder` in the printed graph correspond to function parameters. In this specific lab scenario, the target 'x' is used as a placeholder, representing a function input.

The operation `get_attr` is employed to retrieve specific attributes or parameters from a module, enhancing the analysis of the computational graph.

The term `call_function` is associated with the utilization of standalone, user-defined, or built-in functions within the computation graph. This function operates independently of any object or module, taking inputs and producing an output.

`call_module` encompasses both data (parameters, states) and methods (like the forward method). When this function is invoked, it essentially calls the forward method of the specified module with the given arguments.

Lastly, `call_method` signifies a method call on an object, specifically bound to the tensor object in this context.

# Question 2
`profile_statistics_analysis_pass`:  Execute profile statistics analysis on the provided graph. This involves gathering statistics related to parameters and activations and storing them in the metadata of the respective nodes. Statistics involves the following functions:

#### Record
  - **Description:** Records all input samples.
  - **Arguments:**
    - `device (str|None)`: Specifies the device to move the samples. If None, no device movement occurs.
    - `add_new_dim_before_concat (bool)`: When True, adds a new dimension before concatenating the samples.

#### VarianceOnline
  - **Description:** Utilizes Welford's online algorithm to compute running variance and mean. It conserves memory by not storing all samples, but precision diminishes with smaller counts.
  - **Arguments:**
    - `device (str|None)`: Designates the device for sample movement. If None, no movement takes place.
    - `dims (str|list|None)`: Determines the dimensions to reduce. If "all," reduces all dimensions. If None, no dimension reduction occurs. If a list, reduces the specified dimensions.

#### VariancePrecise
  - **Description:** Concatenates samples and uses `torch.var` and `torch.mean` for precise variance and mean calculation. More memory-intensive but provides accurate results.
  - **Arguments:**
    - `device (str|None)`: Specifies the device for sample movement. If None, no device movement occurs.
    - `dims (str|list|None)`: Determines the dimensions to reduce. If "all," reduces all dimensions. If None, no dimension reduction occurs. If a list, reduces the specified dimensions.

#### RangeNSigma
  - **Description:** Computes the range of samples within n sigma based on mean and variance, assuming normal distribution.
  - **Arguments:**
    - `device (str|None)`: Specifies the device for sample movement. If None, no device movement occurs.
    - `dims (str|list|None)`: Determines the dimensions to reduce. If "all," reduces all dimensions. If None, no dimension reduction occurs. If a list, reduces the specified dimensions.
    - `abs (bool)`: When True, takes the absolute value of samples before calculating mean and variance.
    - `var_mode (str)`: "precise" or "online". If "precise," uses `VariancePrecise` for variance. If "online," uses `VarianceOnline`.
    - `num_sigma (int)`: Specifies the number of sigma to calculate the range.

#### RangeMinMax
  - **Description:** Computes the range of samples based on minimum and maximum values.
  - **Arguments:**
    - `device (str|None)`: Specifies the device for sample movement. If None, no device movement occurs.
    - `dims (str|list|None)`: Determines the dimensions to reduce. If "all," reduces all dimensions. If None, no dimension reduction occurs. If a list, reduces the specified dimensions.
    - `abs (bool)`: When True, takes the absolute value of samples before calculating min and max.

#### RangeQuantile
  - **Description:** Computes the range of samples based on specified quantiles.
  - **Arguments:**
    - `device (str|None)`: Specifies the device for sample movement. If None, no device movement occurs.
    - `dims (str|list|None)`: Determines the dimensions to reduce. If "all," reduces all dimensions. If None, no dimension reduction occurs. If a list, reduces the specified dimensions.
    - `abs (bool)`: When True, takes the absolute value of samples before calculating quantiles.
    - `quantile (float)`: Specifies the quantile used for range calculation.

#### AbsMean
  - **Description:** Implements an online algorithm to compute the mean, considering the absolute value (E(|x|)).
  - **Arguments:**
    - `device (str|None)`: Specifies the device for sample movement. If None, no device movement occurs.
    - `dims (str|list|None)`: Determines the dimensions to reduce. If "all," reduces all dimensions. If None, no dimension reduction occurs. If a list, reduces the specified dimensions.

`report_node_meta_param_analysis_pass`: Conducts a meta-parameter analysis on the graph nodes, including those supplied for the `profile_statistics_analysis_pass`, and produces a comprehensive report based on this examination. The "which" parameter allows the selection of one of three options: "Common", "Hardware", and "Software".


# Question 3
The `quantize_transform_pass` function receives arguments (via pass_args), and specifically, it only takes the linear operator as input. Consequently, during the execution of the pass, only the linear operator undergoes modifications. Since the jsc-tiny model contains just a single linear operator, this implies that only one OP is altered.

# Question 4

# Question 5

# Question 6

# Question 7
