#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include<opencv2/gpu/gpu.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// Computes the x component of the gradient vector
// at a given point in a image.
// returns gradient in the x direction
int xGradient(Mat *image, int x, int y) {
    return image->at<uchar>(y-1, x-1) +
                2*image->at<uchar>(y, x-1) +
                 image->at<uchar>(y+1, x-1) -
                  image->at<uchar>(y-1, x+1) -
                   2*image->at<uchar>(y, x+1) -
                    image->at<uchar>(y+1, x+1);
}

__device__ float xGradientf(float *A, int x, int y, int rows) {
    return A[(y-1)*rows+x-1] +
                2*A[y*rows+x-1] +
                    A[(y+1)*rows+x-1] -
                        A[(y-1)*rows+x+1] -
                            2*A[y*rows+x+1] -
                                A[(y+1)*rows+x+1];
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction
int yGradient(Mat *image, int x, int y) {
    return image->at<uchar>(y-1, x-1) +
                2*image->at<uchar>(y-1, x) +
                 image->at<uchar>(y-1, x+1) -
                  image->at<uchar>(y+1, x-1) -
                   2*image->at<uchar>(y+1, x) -
                    image->at<uchar>(y+1, x+1);
}

__device__ float yGradientf(float *A, int x, int y, int rows) {
    return A[(y-1)*rows+x-1] +
                2*A[(y-1)*rows+x] +
                    A[(y-1)*rows+x+1] -
                        A[(y+1)*rows+x-1] -
                            2*A[(y+1)*rows+x] -
                                A[(y+1)*rows+x+1];

}

__global__ void edges(float *A, float *B, int rows) {
    int y = blockIdx.x+1, x = threadIdx.x+1;
    float gx = xGradientf(A, x, y, rows);
    float gy = yGradientf(A, x, y, rows);
    int sum = sqrtf(gx*gx + gy*gy);
    B[y*rows+x] = fmaxf(0, fminf(sum, 255));
}

int main() {
    Mat src, dst;
    src = imread("obh.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    float *A = (float *) malloc(sizeof(float)*src.rows*src.cols);
    float *B = (float *) malloc(sizeof(float)*src.rows*src.cols);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            A[y*src.rows+x] = src.at<uchar>(y, x);
            B[y*src.rows+x] = 0.0;
        }
    }

    float *d_A, *d_B;
    cudaMalloc((void **)&d_A, sizeof(float)*src.rows*src.cols);
    cudaMalloc((void **)&d_B, sizeof(float)*src.rows*src.cols);
    cudaMemcpy(d_A, A, sizeof(float)*src.rows*src.cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*src.rows*src.cols, cudaMemcpyHostToDevice);
    edges<<<src.rows-1, src.cols-1>>>(d_A, d_B, src.rows);
    cudaMemcpy(B, d_B, sizeof(float)*src.rows*src.cols, cudaMemcpyDeviceToHost);

    dst = src.clone();
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            dst.at<uchar>(y, x) = B[y*src.rows+x];

    namedWindow("final");
    imshow("final", dst);

    namedWindow("initial");
    imshow("initial", src);

    imwrite("obhfinal.jpg",dst);
    waitKey();

    return 0;
}