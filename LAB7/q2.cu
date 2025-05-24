#include <stdio.h>
#include <string.h>
#include <cuda.h>

__global__ void transformKernel(char *S, char *RS, int len) {
    int i = threadIdx.x;

    if (i < len)
        RS[i] = S[i];             // First PCAP
    else if (i < 2 * len)
        RS[i] = S[i - len];       // Second PCAP
    else if (i < 3 * len - 1) {   // Third PCP (omit index 1 of S)
        int idx = i - 2 * len;
        RS[i] = (idx >= 1) ? S[idx + 1] : S[idx];  // Skip second character
    }
}

int main() {
    const char *S = "PCAP";
    int len = strlen(S);
    int newLen = 3 * len - 1;  // PCAP + PCAP + PCP

    char RS[newLen + 1];  // +1 for null terminator

    char *devS, *devRS;

    cudaMalloc((void**)&devS, len + 1);
    cudaMalloc((void**)&devRS, newLen + 1);

    cudaMemcpy(devS, S, len + 1, cudaMemcpyHostToDevice);

    transformKernel<<<1, newLen>>>(devS, devRS, len);
    cudaMemcpy(RS, devRS, newLen, cudaMemcpyDeviceToHost);
    RS[newLen] = '\0';

    printf("Output String RS: %s\n", RS);

    cudaFree(devS);
    cudaFree(devRS);
    return 0;
}

