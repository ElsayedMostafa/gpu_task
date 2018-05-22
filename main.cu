//http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include <time.h>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "stb_image_write.h"

using namespace std;

#define num_threads_per_block 64


inline bool exists(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

// Kernel definition
// Run on GPU
__global__ void histogram_kernel_gray(int size, uint8_t *gray_image, unsigned int *hist_counts) {
    //!histogram
    /*    //!needs lock :( :(  atomicInc doesn't work well :( :(
   int index = blockIdx.x * blockDim.x + threadIdx.x;

      if (index < size){
          hist_counts[gray_image[index]]=atomicInc(&hist_counts[gray_image[index]],(unsigned int)10000000);
      }
      */
    //!histogram
    for (int i = 0; i < size; i++)
        if (gray_image[i] == threadIdx.x)
            hist_counts[threadIdx.x]++;
}

// Kernel definition
// Run on GPU
__global__ void LUT_kernel_gray(uint8_t *LUT, unsigned int *hist_counts, int size) {
    LUT[threadIdx.x] = static_cast<uint8_t>(round(255.0 * hist_counts[threadIdx.x] / (size)));
}

// Kernel definition
// Run on GPU
__global__ void LUT_replacement_gray(uint8_t *LUT, uint8_t *gray_image, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        gray_image[index] = LUT[gray_image[index]];
}

void checkerros(string m = "") {
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("Add Kernel launch failed: %s %s\n", cudaGetErrorString(cudaStatus),m.c_str());
        exit(0);
    }
}

void gpu_histogram_equalization_grayscale(const string &path, const string &filename) {
    if (!exists(path + filename)) {
        cout << "File Doesn't Exist";
        return;
    }

    int width, height, channels;
    int desired_channels = 1;

    unsigned int *hist_counts = new unsigned int[256];
    uint8_t *LUT = new uint8_t[256];

    uint8_t *gray_image = stbi_load((path + filename).c_str(), &width, &height, &channels, desired_channels);
    cout << "image read : " << width << " " << height << " " << channels<<endl;

    int image_size = width * height * desired_channels;

    //! START : histogram can be parallelized
    //! define
    uint8_t *d_gray_image;
    unsigned int *d_hist_counts;

    //!Alocate and transfer data
    cudaMalloc((void **) &d_gray_image, image_size * sizeof(uint8_t));
    cudaMemcpy(d_gray_image, gray_image, image_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_hist_counts, 256 * sizeof(int));

    //! Launch histogram_kernel_gray() kernel on GPU
    histogram_kernel_gray << < 1, 256 >> > (image_size, d_gray_image, d_hist_counts);

    // Check for any errors launching the kernel
    checkerros("histogram_kernel_gray");

    //! wait
    cudaDeviceSynchronize();
    cout << "Done:histogram_kernel_gray"<<endl;

    //!Transfer P from device to host
    cudaMemcpy(hist_counts, d_hist_counts, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //!Free device matrices
    cudaFree(d_gray_image);
    cudaFree(d_hist_counts);
    //! END : histogram can be parallelized

    //! START : CDF can not be parallelized
    //!CDF
    for (int i = 1; i < 256; i++)
        hist_counts[i] += hist_counts[i - 1];

    //cout << "Final value of CDF: " << hist_counts[255] << endl;
    //! END : CDF can not be parallelized

    //! START : LUT computation can be parallelized
    //!LUT
    //! define
    uint8_t *d_LUT;
    cudaMalloc((void **) &d_hist_counts, 256 * sizeof(int));
    cudaMemcpy(d_hist_counts, hist_counts, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_LUT, 256 * sizeof(int));

    LUT_kernel_gray << < 1, 256 >> > (d_LUT, d_hist_counts, width * height);
    checkerros("LUT_kernel_gray");

    //! wait
    cudaDeviceSynchronize();
    cout << "Done:LUT_kernel_gray"<<endl;

    //!Transfer LUT from device to host
    cudaMemcpy(LUT, d_LUT, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //!Free device matrices
    //! cudaFree(d_LUT); free it next
    cudaFree(d_hist_counts);
    //! END : LUT computation be parallelized


    int num_blocks = ((image_size + num_threads_per_block - 1) / num_threads_per_block);
    cout << "\nnumber of blocks " << num_blocks << " each one has ("<<num_threads_per_block<<" threads)" << endl;

    cudaMalloc((void **) &d_gray_image, image_size * sizeof(uint8_t));
    cudaMemcpy(d_gray_image, gray_image, image_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    //! START : replacement using LUT can be parallelized
    LUT_replacement_gray << < num_blocks, num_threads_per_block >> > (d_LUT, d_gray_image, image_size);
    checkerros("LUT_replacement_gray");

    //! wait
    cudaDeviceSynchronize();
    cout << "Done:LUT_replacement_gray"<<endl;

    //!Transfer equalized image from device to host
    cudaMemcpy(gray_image, d_gray_image, image_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_LUT);
    cudaFree(d_gray_image);
    //! END : replacement using LUT can be parallelized

    stbi_write_jpg((path + "gpu_equ_" + filename).c_str(), width, height, 1, gray_image, 1000);

    //! free resources
    stbi_image_free(gray_image);

    delete[] hist_counts;
    delete[] LUT;
}

// Kernel definition
// Run on GPU
__global__ void histogram_kernel_rgb(int size, uint8_t *gray_image, unsigned int *hist_counts) {
    //!histogram
    for (int i = blockIdx.x; i < size; i+=3)//!channels
        if (gray_image[i] == threadIdx.x)
            hist_counts[threadIdx.x + blockIdx.x*256]++;
}

// Kernel definition
// Run on GPU
__global__ void LUT_kernel_rgb(uint8_t *LUT, unsigned int *hist_counts, int size) {
    LUT[blockIdx.x*256+threadIdx.x] = static_cast<uint8_t>(round(255.0 * hist_counts[blockIdx.x*256+threadIdx.x] / (size)));
}

// Kernel definition
// Run on GPU
__global__ void LUT_replacement_rgb(uint8_t *LUT, uint8_t *gray_image, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        gray_image[index] = LUT[gray_image[index]+(index%3)*256];
}


void gpu_histogram_equalization_rgb(const string &path, const string &filename) {
    if (!exists(path + filename)) {
        cout << "File Doesn't Exist";
        return;
    }
    int width, height, channels;
    int desired_channels = 3;

    unsigned int *hist_counts = new unsigned int[256*3];
    uint8_t *LUT = new uint8_t[256*3];

    uint8_t *gray_image = stbi_load((path + filename).c_str(), &width, &height, &channels, desired_channels);
    cout << "image read : " << width << " " << height << " " << channels<<endl;

    int image_size = width * height * desired_channels;
    //! START : histogram can be parallelized
    //! define
    uint8_t *d_gray_image;
    unsigned int *d_hist_counts;

    //!Alocate and transfer data
    cudaMalloc((void **) &d_gray_image, image_size * sizeof(uint8_t));

    cudaMemcpy(d_gray_image, gray_image, image_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_hist_counts, 3*256 * sizeof(int));

    //! Launch histogram_kernel_gray() kernel on GPU
    //int num_blocks = ((width * height * desired_channels + num_threads_per_block - 1) / num_threads_per_block);
    //cout << "\nnumber of blocks " << num_blocks << " each one has (1024 threads) " << endl;

    histogram_kernel_rgb << < 3, 256 >> > (image_size, d_gray_image, d_hist_counts);

    // Check for any errors launching the kernel
    checkerros("histogram_kernel_rgb");

    //! wait
    cudaDeviceSynchronize();
    cout << "Done:histogram_kernel_rgb"<<endl;
    //!Transfer P from device to host
    cudaMemcpy(hist_counts, d_hist_counts, 3*256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //!Free device matrices
    //cudaFree(d_gray_image); later
    cudaFree(d_hist_counts);
    //! END : histogram can be parallelized

    //! START : CDF can not be parallelized
    //!CDF
    for (int i = 1; i < 256; i++)
        for (int channel = 0; channel < desired_channels; ++channel)
            hist_counts[i+256*channel] += hist_counts[i - 1+256*channel];

    //cout << "Final value of CDF: " << hist_counts[255] << endl;
    //! END : CDF can not be parallelized

    //! START : LUT computation can be parallelized
    //!LUT
    //! define
    uint8_t *d_LUT;
    cudaMalloc((void **) &d_hist_counts, 3*256 * sizeof(int));
    cudaMemcpy(d_hist_counts, hist_counts, 3*256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_LUT, 3*256 * sizeof(int));

    LUT_kernel_rgb << < 3, 256 >> > (d_LUT, d_hist_counts, width * height);
    checkerros("LUT_kernel_rgb");

    //! wait
    cudaDeviceSynchronize();
    cout << "Done:LUT_kernel_rgb"<<endl;

    //!Transfer LUT from device to host
    cudaMemcpy(LUT, d_LUT, 3*256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //!Free device matrices
    //! cudaFree(d_LUT); free it next
    cudaFree(d_hist_counts);
    //! END : LUT computation be parallelized


    int num_blocks = ((image_size + num_threads_per_block - 1) / num_threads_per_block);
    //cout << "\nnumber of blocks " << num_blocks << " each one has (1024 threads)" << endl;

    //! START : replacement using LUT can be parallelized
    LUT_replacement_rgb << < num_blocks, num_threads_per_block >> > (d_LUT, d_gray_image, image_size);
    checkerros("LUT_replacement_rgb");

    //! wait
    cudaDeviceSynchronize();
    cout << "Done:LUT_replacement_rgb"<<endl;

    //!Transfer equalized image from device to host
    cudaMemcpy(gray_image, d_gray_image, image_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_LUT);
    cudaFree(d_gray_image);
    //! END : replacement using LUT can be parallelized

    stbi_write_jpg((path + "gpu_equ_" + filename).c_str(), width, height, desired_channels, gray_image, 1000);

    //! free resources
    stbi_image_free(gray_image);

    delete[] hist_counts;
    delete[] LUT;
}

void cpu_histogram_equalization_grayscale(const string &path, const string &filename) {
    if (!exists(path + filename)) {
        cout << "File Doesn't Exist";
        return;
    }
    int width, height, channels;
    int desired_channels = 1;

    int hist_counts[256];
    uint8_t LUT[256];
    memset(hist_counts, 0, sizeof hist_counts);
    memset(LUT, 0, sizeof LUT);

    uint8_t *gray_image = stbi_load((path + filename).c_str(), &width, &height, &channels, desired_channels);
    cout << "image read : " << width << " " << height << " " << channels<<endl;

    //!histogram
    for (int i = 0; i < width * height * desired_channels; i++)
        hist_counts[gray_image[i]]++;

/*    for (int i = 0; i < 256; i++)
        cout << hist_counts[i] << ",";
    cout << endl;*/

    //!CDF
    for (int i = 1; i < 256; i++)
        hist_counts[i] += hist_counts[i - 1];

    //cout << "Final value of CDF: " << hist_counts[255] << endl;

    //!LUT
    for (int i = 0; i < 256; i++)
        LUT[i] = static_cast<uint8_t>(round(255.0 * hist_counts[i] / (width * height)));



    //!from LUT
    for (int i = 0; i < width * height * desired_channels; i++)
        gray_image[i] = LUT[gray_image[i]];


    stbi_write_jpg((path + "cpu_equ_" + filename).c_str(), width, height, 1, gray_image, 1000);

    stbi_image_free(gray_image);
}

void cpu_histogram_equalization_rgb(const string &path, const string &filename) {
    if (!exists(path + filename)) {
        cout << "File Doesn't Exist";
        return;
    }
    int width, height, channels;
    int desired_channels = 3;

    int hist_counts[3][256];
    uint8_t LUT[3][256];
    memset(hist_counts, 0, sizeof hist_counts);
    memset(LUT, 0, sizeof LUT);

    uint8_t *rgb_image = stbi_load((path + filename).c_str(), &width, &height, &channels, desired_channels);
    cout << "image read : " << width << " " << height << " " << channels<<endl;

    //!histogram
    for (int i = 0; i < width * height * desired_channels; i++)
        hist_counts[i % 3][rgb_image[i]]++;

    //!CDF
    for (int i = 1; i < 256; i++)
        for (int channel = 0; channel < desired_channels; ++channel)
            hist_counts[channel][i] += hist_counts[channel][i - 1];

    //!LUT
    for (int i = 0; i < 256; i++)
        for (int channel = 0; channel < desired_channels; ++channel)
            LUT[channel][i] = static_cast<uint8_t>(round(255.0 * hist_counts[channel][i] / (width * height)));

    //!from LUT
    for (int i = 0; i < width * height * desired_channels; i += desired_channels)
        for (int channel = 0; channel < desired_channels; ++channel)
            rgb_image[i + channel] = LUT[channel][rgb_image[i + channel]];

    stbi_write_jpg((path + "cpu_equ_" + filename).c_str(), width, height, desired_channels, rgb_image, 1000);

    stbi_image_free(rgb_image);
}

int main(int argc, char **argv) {
    cout << argv[2]<<endl;

    if (strcmp(argv[1], "gg") == 0)
        gpu_histogram_equalization_grayscale("./images/", string(argv[2]));
    if (strcmp(argv[1], "cg") == 0)
        cpu_histogram_equalization_grayscale("./images/",string(argv[2]) );
    if (strcmp(argv[1], "crgb") == 0)
        cpu_histogram_equalization_rgb("./images/", string(argv[2]));
    if (strcmp(argv[1], "grgb") == 0){
        gpu_histogram_equalization_rgb("./images/", string(argv[2]));
    }

    return 0;
}



