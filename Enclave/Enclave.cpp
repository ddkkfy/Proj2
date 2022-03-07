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
    int row1 = *dimW, col1 = *(dimW + 1);
    int row2 = *dimInp, col2 = *(dimInp + 1);

    float* weight = new float[row1 * col1];
    float* input = new float[row2 * col2];
    float* result = new float[row1 * col2];

    memcpy(weight, w, sizeof(w));
    memcpy(input, inp, sizeof(inp));
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            float temp = 0;
            for (int k = 0; k < col1; k++) {
                float left = *(weight + i * col1 + k);
                float right = *(input + k * col2 + j);
                temp += left * right;
                //temp += weight[i][k]*input[k][j]
            }
            *(result + i * col2 + j) = temp;
        }
    }
    memcpy(out, result, sizeof(result));

    delete[]weight;
    delete[]input;
    delete[]result;
}

float* pre;   //new in precompute, delete in removeNoise
float* r;

void ecall_precompute(float* weight, int* dim, int batch) {
    int row = *dim, col = *(dim + 1);
    //size of r is col * batch

    pre = new float[row * batch];
    r = new float[col * batch];
    for (int t = 0; t < col*batch; t++)
        sgx_read_rand((uint8_t*)(r + t), 4);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < batch; j++) {
            float temp = 0;
            for (int k = 0; k < col; k++) {
                float left = *(weight + i * col + k);
                float right = *(r + k * batch + j);
                temp += left * right;
                //temp += weight[i][k]*r[k][j]
            }
            *(pre + i * batch + j) = temp;
        }
    }
}

void ecall_addNoise(float* inp, int* dim, float* out) {
    int row = *dim, col = *(dim + 1);
    float* input = new float[row * col];
    float* result = new float[row * col];

    memcpy(input, inp, sizeof(inp));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float left = *(input + i * col + j);
            float right = *(r + i * col + j);
            *(result + i * col + j) = left + right;
        }
    }
    memcpy(out, result, sizeof(result));

    delete[]input;
    delete[]result;
}

void ecall_removeNoise(float* inp, int* dim, float* out) {
    int row = *dim, col = *(dim + 1);
    float* input = new float[row * col];
    float* result = new float[row * col];

    memcpy(input, inp, sizeof(inp));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float left = *(input + i * col + j);
            float right = *(pre + i * col + j);
            *(result + i * col + j) = left - right;
        }
    }
    memcpy(out, result, sizeof(result));

    delete[]input;
    delete[]result;
    delete[]r;
    delete[]pre;
}