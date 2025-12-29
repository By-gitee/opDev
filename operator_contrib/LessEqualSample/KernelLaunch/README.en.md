## Overview
This example demonstrates the direct kernel function invocation method for the LessEqual operator.

## Directory Structure Introduction
``` 
├── KernelLaunch                      // Invokes the LessEqual custom operator directly using the kernel function
│   └── LessEqualKernelInvocation           // Host-side kernel function invocation program
``` 

## Compiling and Running the Example Operator
For custom operator projects, the compilation and running process includes the following steps:
- Compile the custom operator project;
- Invoke and execute the custom operator;

Detailed operations are as follows.

### 1. Obtain the Source Code Package
Before compiling and running this example, please refer to [Preparation: Obtain the Example Code](../README.en.md#codeready) to obtain the source code package.

### 2. Compile and Run the Example Project
- [LessEqualKernelInvocation Example Execution](./LessEqualKernelInvocation/README.en.md)

## Update Log
| Date       | Update Item | Notes |
|------------|-------------|-------|
| 2024/05/24 | Updated readme structure | Requires running on community CANN package version 7.0.0.alpha003 or later |