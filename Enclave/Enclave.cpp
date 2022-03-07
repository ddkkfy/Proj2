#include "Enclave_t.h" /* print_string */

#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>    // std::max
#include <sgx_trts.h>   //sgx_read_rand


int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}


// the actual buffer of *inp is in untrusted memory
// You can read from it, but never write to it
int ecall_compute_secrete_operation(int* inp, int size) {
    // decrypt inp
    // ....

    int res = 0;

    for (int i = 0; i < size; i++) {
        res += inp[i];
    }

    // encrypt res
    // ....

    printf("Returning to App.cpp\n");
    return res;
}

void ecall_nativeMatMul(float* w, int* dimW, float* inp, int* dimInp, float* out) {
    int row1 = *dimInp, col1 = *(dimInp + 1);
    int row2 = *dimW, col2 = *(dimW + 1);

    float* weight = new float[row2 * col2];
    float* input = new float[row1 * col1];
    float* result = new float[row1 * col2];

    memcpy(weight, w, sizeof(float)*row2*col2);
    memcpy(input, inp, sizeof(float)*row1*col1);
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            float temp = 0;
            for (int k = 0; k < col1; k++) {
                float left = *(input + i * col1 + k);
                float right = *(weight + k * col2 + j);
                temp += left * right;
                //temp += input[i][k]*weight[k][j]
            }
            *(result + i * col2 + j) = temp;
        }
    }
    //printf("Enclave native: %f\n", *(result+1));
    memcpy(out, result, sizeof(float)*row1*col2);

    delete[]weight;
    delete[]input;
    delete[]result;
}

float* pre;   //new in precompute, delete in removeNoise
float* r;

void ecall_precompute(float* weight, int* dim, int batch) {
    int row = *dim, col = *(dim + 1);
    //size of r is batch * row

    pre = new float[batch * col];
    r = new float[batch * row];
    float* w = new float[row * col];
    memcpy(w, weight, sizeof(float)*row*col);
    
    for (int t = 0; t < batch*row; t++)
        sgx_read_rand((uint8_t*)(r + t), 4);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < col; j++) {
            float temp = 0;
            for (int k = 0; k < row; k++) {
                float left = *(r + i * row + k);
                float right = *(w + k * col + j);
                temp += left * right;
                //temp += r[i][k]*weight[k][j]
            }
            *(pre + i * col + j) = temp;
        }
    }
}

void ecall_addNoise(float* inp, int* dim, float* out) {
    int row = *dim, col = *(dim + 1);
    float* input = new float[row * col];
    float* result = new float[row * col];

    memcpy(input, inp, sizeof(float)*row*col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float left = *(input + i * col + j);
            float right = *(r + i * col + j);
            *(result + i * col + j) = left + right;
        }
    }
    memcpy(out, result, sizeof(float)*row*col);

    delete[]input;
    delete[]result;
}

void ecall_removeNoise(float* inp, int* dim, float* out) {
    int row = *dim, col = *(dim + 1);
    float* input = new float[row * col];
    float* result = new float[row * col];

    memcpy(input, inp, sizeof(float)*row*col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float left = *(input + i * col + j);
            float right = *(pre + i * col + j);
            *(result + i * col + j) = left - right;
        }
    }

    printf("Enclave method [1][1]: %f\n", *(result + col + 1));
    memcpy(out, result, sizeof(float)*row*col);

    delete[]input;
    delete[]result;
    delete[]r;
    delete[]pre;
}