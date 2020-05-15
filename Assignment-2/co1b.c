#include <stdio.h>
#include <stdlib.h>
#define n 10*1024*1024
struct DATA{int a;int b;};
struct DATA pMyData[n];int main(){for (long i=0; i<10*1024*1024; i++){pMyData[i].a = pMyData[i].b;}return 0;}