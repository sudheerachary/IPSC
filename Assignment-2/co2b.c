#include <stdio.h>
#include <stdlib.h>
#define nRows 1024*8
#define nCols 1024*8
char MyData[nRows*nCols];int main(){for (long y=0; y < nCols; y++)for (long x=0; x < nRows; x++){MyData[x+y*nRows]++;}return 0;}