# `Using Buffer Args` Sample

## Purpose

To learn how to pass sycl buffers as arguments

## Reference
This sample was modified after the *[oneAPI Samples Catalog](https://oneapi-src.github.io/oneAPI-samples/)*.
Specifically the DirectProgramming->DenseLinearAlgebra->simple-add example. 

This readme is largely the same as the above oneAPI example.

## Prerequisites

Warning: Unbuntu should work, but it was not tested. Only Windows 10 with vscode was used. The USM
example is also not completed.

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04 <br> Windows* 10   (this sample not tried on Ubuntu)
| Software           | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

This sample provides examples of both buffers and USM implementations (not done) for simple side-by-side comparison.

- USM requires an explicit wait for the asynchronous kernel's
computation to complete.
- Buffers, at the time they go out of scope, bring main
memory in sync with device memory implicitly. The explicit wait on the event is
not required as a result.

> **Note**: For comprehensive information about oneAPI programming, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:  (warning: this sample was not tried on Ubuntu)
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

### Using Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
3. Open a terminal in VS Code (**Terminal > New Terminal**).
4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
*[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

### On Linux*

#### Configure the build system

1. Change to the sample directory.
2.
   Configure the project to use the buffer-based implementation.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   or

   Configure the project to use the Unified Shared Memory (USM) based implementation.
   ```
   mkdir build
   cd build
   cmake .. -DUSM=1
   ```

   > **Note**: When building for FPGAs, the default FPGA family will be used (Intel® Agilex® 7).
   > You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

#### Build for CPU and GPU

1. Build the program.
   ```
   make cpu-gpu
   ```
2. Clean the program. (Optional)
   ```
   make clean
   ```

### On Windows*

#### Configure the build system

1. Change to the sample directory.
2.
   Configure the project to use the buffer-based implementation.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   or

   Configure the project to use the Unified Shared Memory (USM) based implementation.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" .. -DUSM=1
   ```

   > **Note**: When building for FPGAs, the default FPGA family will be used (Intel® Agilex® 7).
   > You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

#### Build for CPU and GPU

1. Build the program.
   ```
   nmake cpu-gpu
   ```
2. Clean the program. (Optional)
   ```
   nmake clean
   ```
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Run the `Simple Add` Program

### On Linux

#### Run for CPU and GPU

1. Change to the output directory.

2. Run the program for Unified Shared Memory (USM) and buffers.
   ```
   ./gaussian-buffers
   ./gaussian-usm
   ```
### On Windows

#### Run for CPU and GPU

1. Change to the output directory.

2. Run the program for Unified Shared Memory (USM) and buffers.
   ```
   Visual Studio Powershell window
   .\syclBufferArgs.exe
   ```
## Example Output
oneApiPassBufferToFunction\build> .\syclBufferArgs.exe
Starting main
Running on device: Intel(R) UHD Graphics
Error u8_data[0] = 0 does not equal init value 128.
