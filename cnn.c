#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "cnn.h"

#include <string.h>
#include <CL/cl.h>
#include "kernel_cl.h"

#define LOCAL_SIZE 16
#define GLOBAL_SIZE 32

#define H 8
#define W 32

struct timeval t1, t2, t3;

inline void checkErr(cl_int err, const char * name) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: %s (%d)\n", name, err);
    exit(EXIT_FAILURE);
  }
}

void opencl_gpu(
   float Cout[NUM][OUTIMROW][OUTIMROW],
   float Cin[NUM][INIMROW][INIMROW],
   float weight[NUM][NUM][KERNEL][KERNEL],
   float bias[NUM]
){
  // please add your OpenCL setup code below
  // Use this to check the output of each API call                                                                                                             
  cl_int status;
  int i;
  
  // Retrieve the number of platforms                                                                                                                          
  cl_uint numPlatforms = 0;
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  checkErr(status, "Retrieve the number of platforms");

  // Allocate enough space for each platform                                                                                                                   
  cl_platform_id *platforms = NULL;
  platforms = (cl_platform_id*)malloc(
                                      numPlatforms * sizeof(cl_platform_id));

  // Fill in the platforms                                                                                                                                     
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  checkErr(status, "Fill in the platforms");

  // Find CPU                                                                                                                                                  
  int platform_index = -1;
  for (i = 0; i < numPlatforms; i++){
    char vendor[128];
    clGetPlatformInfo (platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    char vendorF[7];
    memcpy((void*)vendorF, (void*)vendor, 6);
    vendorF[6] = '\0';
    fprintf(stderr, "%s\n", vendorF);
    if (strcmp(vendorF, "NVIDIA") == 0)
      {
        platform_index = i;
        break;
      }
  }
  if (platform_index == -1){
    printf("GPU platform not found!\n");
    exit(1);
  }

  // Retrieve the number of devices                                                                                                                            
  cl_uint numDevices = 0;
  status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0,
                          NULL, &numDevices);
  checkErr(status, "Retrieve the number of devices");
  printf("#devices: %d, status %d\n", numDevices, status);

  cl_device_id *devices;
  devices = (cl_device_id*)malloc(
                                  numDevices * sizeof(cl_device_id));

  // Fill in the devices                                                                                                                                       
  status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL,
                          numDevices, devices, NULL);
  checkErr(status, "Fill in the devices");

  gettimeofday(&t1, NULL);
  
  // Create a context and associate it with the devices                                                                                                        
  cl_context context;
  context = clCreateContext(NULL, numDevices, devices, NULL,
                            NULL, &status);

  // Create a command queue and associate it with the device                                                                                                   
  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueue(context, devices[0], 0,
                                  &status);
  
  // Create a buffer object that will contain the data
  // from the host array A
  cl_mem buf_cout, buf_cin, buf_weight, buf_bias, buf_C;

  int cout_size = sizeof(float)*NUM*OUTIMROW*OUTIMROW;
  int cin_size = sizeof(float)*NUM*INIMROW*INIMROW;
  int bias_size = sizeof(float)*NUM;
  int weight_size = sizeof(float)*NUM*NUM*KERNEL*KERNEL;
  int C_size = sizeof(float)*NUM*IMROW*IMROW;

  buf_cout = clCreateBuffer(context, CL_MEM_WRITE_ONLY, cout_size, NULL, &status);
  buf_cin = clCreateBuffer(context, CL_MEM_READ_ONLY, cin_size, NULL, &status);
  buf_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, NULL, &status);
  buf_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, &status);
  buf_C = clCreateBuffer(context, CL_MEM_READ_WRITE, C_size, NULL, &status);

  status = clEnqueueWriteBuffer(cmdQueue, buf_cin, CL_FALSE,
                                0, cin_size, Cin, 0, NULL, NULL);
  checkErr(status, "Write buffer Cin");


  status = clEnqueueWriteBuffer(cmdQueue, buf_weight, CL_FALSE,
                                0, weight_size, weight, 0, NULL, NULL);
  checkErr(status, "Write buffer weight");

  status = clEnqueueWriteBuffer(cmdQueue, buf_bias, CL_FALSE,
                                0, bias_size, bias, 0, NULL, NULL);
  checkErr(status, "Write buffer bias");

  // Create a program with source code
  cl_program program = clCreateProgramWithSource(context, 1,
                                                 (const char**)&kernel_cl, NULL, &status);
  
  // Build (compile) the program for the device
  status = clBuildProgram(program, numDevices, devices,
                          NULL, NULL, NULL);
  if(status == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &log_size);
    char *log = (char*)malloc(log_size);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
                          log_size, log, NULL);
    fprintf(stderr, "%s\n", log);
    exit(1);
  }
  checkErr(status, "Build program");

  // Create the vector addition kernel
  cl_kernel kernel[4];
  kernel[0] = clCreateKernel(program, "get_bias", &status);
  kernel[1] = clCreateKernel(program, "convolution", &status);
  kernel[2] = clCreateKernel(program, "ReLU", &status);
  kernel[3] = clCreateKernel(program, "max_pooling", &status);

  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  gettimeofday(&t3, NULL);
  //kernel 0
  status = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &buf_C);
  checkErr(status, "kernel 0 Set Arg 0");
  status = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &buf_bias);
  checkErr(status, "kernel 0 Set Arg 1");

  globalWorkSize[0] = NUM;
  globalWorkSize[1] = IMROW;
  globalWorkSize[2] = IMROW;
  
  localWorkSize[0] = 1;
  localWorkSize[1] = 1;
  localWorkSize[2] = 32;
        
  status = clEnqueueNDRangeKernel(cmdQueue, kernel[0], 3, NULL, 
                                  globalWorkSize, localWorkSize, 0, NULL, NULL);
  checkErr(status, "Execute kernel[0]: get_bias");


  //kernel 1
  status = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &buf_C);
  checkErr(status, "kernel 1 Set Arg 0");
  status = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &buf_weight);
  checkErr(status, "kernel 1 Set Arg 1");
  status = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &buf_cin);
  checkErr(status, "kernel 1 Set Arg 2");


  globalWorkSize[0] = NUM;
  globalWorkSize[1] = IMROW;
  globalWorkSize[2] = IMROW;

  localWorkSize[0] = 1;
  localWorkSize[1] = H;
  localWorkSize[2] = W;
  
  status = clEnqueueNDRangeKernel(cmdQueue, kernel[1], 3, NULL,
                                  globalWorkSize, localWorkSize, 0, NULL, NULL);
  checkErr(status, "Execute kernel[1]: convolution");


  //kernel 2
  status = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &buf_C);
  checkErr(status, "kernel 2 Set Arg 0");

  globalWorkSize[0] = NUM;
  globalWorkSize[1] = IMROW;
  globalWorkSize[2] = IMROW;
        

  localWorkSize[0] = 1;
  localWorkSize[1] = 1;
  localWorkSize[2] = 32;


  status = clEnqueueNDRangeKernel(cmdQueue, kernel[2], 3, NULL,
                                  globalWorkSize, localWorkSize, 0, NULL, NULL);
  checkErr(status, "Execute kernel[2]: ReLU");

  //kernel 3
  status = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &buf_C);
  checkErr(status, "kernel 3 Set Arg 0");
  status = clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &buf_cout);
  checkErr(status, "kernel 3 Set Arg 1");



  globalWorkSize[0] = NUM;
  globalWorkSize[1] = OUTIMROW;
  globalWorkSize[2] = OUTIMROW;


  localWorkSize[0] = 1;
  localWorkSize[1] = 1;
  localWorkSize[2] = 16;


  status = clEnqueueNDRangeKernel(cmdQueue, kernel[3], 3, NULL,
                                  globalWorkSize, localWorkSize, 0, NULL, NULL);
  checkErr(status, "Execute kernel[3]: max_pooling");

  // Read the device output buffer to the host output array
  //clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, 
  //                    datasize, C, 0, NULL, NULL);

  clEnqueueReadBuffer(cmdQueue, buf_cout, CL_TRUE, 0, 
                      cout_size, Cout, 0, NULL, NULL);
}

int main(){
	static float Cout[NUM][OUTIMROW][OUTIMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	fprintf(stderr, "Start cnn computation\n");

	// --- Please add OpenCL setup code inside the function below ---
   opencl_gpu(
      Cout, Cin, weight, bias
   );	

   // --- Timing stuff
	gettimeofday(&t2, NULL);
	float elapsed_time1 = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
	float elapsed_time2 = (t2.tv_sec - t3.tv_sec) + (t2.tv_usec - t3.tv_usec) / 1e6;
	fprintf(stderr, "time(s): %f\n", elapsed_time1);
	fprintf(stderr, "kernel(s): %f\n", elapsed_time2);
	fprintf(stderr, "GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time1 / 1e9);

   // Please disable the error check before handing in your submission
   // Reminder: We will be measuring your performance externally! (using a unix time invocation)
	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	return 0;
}
