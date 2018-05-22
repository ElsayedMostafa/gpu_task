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


inline bool exists(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
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

    //!histogram
    for (int i = 0; i < width * height * desired_channels; i++)
        hist_counts[gray_image[i]]++;

/*    for (int i = 0; i < 256; i++)
        cout << hist_counts[i] << ",";
    cout << endl;*/

    //!CDF
    for (int i = 1; i < 256; i++)
        hist_counts[i] += hist_counts[i - 1];

    cout << "Final value of CDF: " << hist_counts[255] << endl;

    //!LUT
    for (int i = 0; i < 256; i++)
        LUT[i] = static_cast<uint8_t>(round(255.0 * hist_counts[i] / (width * height)));

    cout << "image read : " << width << " " << height << " " << channels<<endl;

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

    cout << "image read : " << width << " " << height << " " << channels<<endl;

    //!from LUT
    for (int i = 0; i < width * height * desired_channels; i += desired_channels)
        for (int channel = 0; channel < desired_channels; ++channel)
            rgb_image[i + channel] = LUT[channel][rgb_image[i + channel]];

    stbi_write_jpg((path + "cpu_equ_" + filename).c_str(), width, height, desired_channels, rgb_image, 1000);

    stbi_image_free(rgb_image);
}

int main(int argc, char **argv) {
    cpu_histogram_equalization_grayscale("./images/", "in-grayscale.jpg");
    cpu_histogram_equalization_rgb("./images/", "in-color.jpg");
    return 0;
}



