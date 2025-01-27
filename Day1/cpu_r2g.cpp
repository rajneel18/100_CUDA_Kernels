#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

void convertToGreyscaleCPU(const unsigned char* h_input, unsigned char* h_output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            unsigned char r = h_input[idx];
            unsigned char g = h_input[idx + 1];
            unsigned char b = h_input[idx + 2];
            h_output[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

int main() {
    string inputImagePath = "input_image.jpg";
    string outputImagePath = "output_cpu_image.jpg";

    Mat image = imread(inputImagePath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not open the image file." << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    unsigned char* h_input = image.data;
    unsigned char* h_output = new unsigned char[width * height];

    // Measure start time
    auto start = high_resolution_clock::now();

    convertToGreyscaleCPU(h_input, h_output, width, height);

    // Measure end time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "CPU processing time: " << duration.count() << " milliseconds." << endl;

    Mat outputImage(height, width, CV_8UC1, h_output);
    imwrite(outputImagePath, outputImage);

    delete[] h_output;

    return 0;
}
