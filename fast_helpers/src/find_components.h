#pragma once

#include <string.h>
#include <stdio.h>

// 50K will use ~782KB in RAM, good and safe enough
#define MAX_COMPONENTS 50000 
// The following variables keep the current min and max coordinates
// on the following format: max_coord = xyxyxyxy....
extern int max_coord[MAX_COMPONENTS*2];
extern int min_coord[MAX_COMPONENTS*2];

extern "C" {
    void find_boxes(const int *markers, int width, int height, int lenOutBoxes, int* outBoxes);
}
