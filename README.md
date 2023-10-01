# gplearn_cross_factor

Welcome to the gplearn_cross_factor project! The primary objective of this project is to enhance the existing gplearn package and enable three-dimensional structured dimension genetic programming (GP) specifically for cross-sectional factor investigation.

## Project Overview

In the initial version, the gplearn package only supported rankIC as the fitness metric. However, with this update, we have introduced significant improvements to extend its functionality and empower factor analysis. The key enhancements in this version include:

### Expanded Fitness Metrics
We have incorporated a range of additional fitness metrics to complement rankIC. These new metrics include irir, quantile returns and monotonicity. By incorporating these metrics, you can conduct a more comprehensive evaluation of factors, leading to improved GP performance.

### Enhanced Base Operators
The modified package now offers an expanded set of base operators, including both time series and cross-sectional capabilities. This enhancement provides increased flexibility, empowering researchers and practitioners to conduct more effective cross-sectional factor analysis. More operators (ts_residual, ts_cov, normalization, standardization, etc.) are coming.

## Getting Started

To get started with gplearn_cross_factor, follow these steps:

1. Import the necessary modules and functions required for factor analysis.
2. Specify your desired fitness metrics, including rankIC, quantile returns, monotonicity, and correlation within factors.
3. Define and prepare your data inputs for analysis.
4. Run the genetic programming algorithm using the provided functions and operators.
5. Evaluate and interpret the results to gain insights into your cross-sectional factors.

For more detailed instructions and examples, please refer to [the documentation](/Functional%20Demo.ipynb) provided in the repository.

## Contributing

We welcome contributions from the community to further enhance and expand the functionalities of gplearn for cross-sectional factor analysis. If you have any ideas, bug reports, or feature requests, feel free to open an issue or submit a pull request.

## Stay Updated

Stay tuned for future releases as I continue to improve and enrich gplearn_cross_factor. Don't miss out on the latest updates and enhancements by watching this repository.

Happy factor analysis with gplearn_cross_factor!
