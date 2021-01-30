Space functions are feature extraction programs that accept images as arguments and return feature
vectors to standard out in the following format:
    [feature1=value1,feature2=value2,...,featureN=valueN]

A well formed space function should always return the same size vector for any valid input

Space functions can be developed by using the Parser.java application in the Parser directory.
Parser intakes a features.properties file which specifies individual feature extraction scripts.
For more information, view the README in the Parser/ directory

For an example on how to build a space function, view the example in the SpaceFunctionCorn/ directory.

