#include <stdio.h>
#include <string.h>
#include <cuda.h>

__device__ int isWordMatch(const char *sentence, int start, int end, const char *target, int targetLen) {
    for (int i = 0; i < targetLen; ++i) {
        if ((start + i >= end) || sentence[start + i] != target[i])
            return 0;
    }
    // Ensure the next char is a space or end
    if (sentence[start + targetLen] != ' ' && sentence[start + targetLen] != '\0')
        return 0;
    return 1;
}

__global__ void countWordKernel(char *sentence, char *target, int *count, int sentenceLen, int targetLen) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sentenceLen)
        return;

    // Check if this thread is at the start of a word
    if (i == 0 || sentence[i - 1] == ' ') {
        if (isWordMatch(sentence, i, sentenceLen, target, targetLen)) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    const char *hostSentence = "hello world hello CUDA hello GPU";
    const char *targetWord = "hello";
    int count = 0;

    int sentenceLen = strlen(hostSentence);
    int targetLen = strlen(targetWord);

    char *devSentence, *devTarget;
    int *devCount;

    cudaMalloc((void**)&devSentence, sentenceLen + 1);
    cudaMalloc((void**)&devTarget, targetLen + 1);
    cudaMalloc((void**)&devCount, sizeof(int));

    cudaMemcpy(devSentence, hostSentence, sentenceLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(devTarget, targetWord, targetLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(devCount, &count, sizeof(int), cudaMemcpyHostToDevice);

    countWordKernel<<<1, sentenceLen>>>(devSentence, devTarget, devCount, sentenceLen, targetLen);

    cudaMemcpy(&count, devCount, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Word '%s' found %d times.\n", targetWord, count);

    cudaFree(devSentence);
    cudaFree(devTarget);
    cudaFree(devCount);
    return 0;
}

