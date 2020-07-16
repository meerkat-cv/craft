#include "find_components.h"

int max_coord[MAX_COMPONENTS*2];
int min_coord[MAX_COMPONENTS*2];

void find_boxes(const int *markers, int width, int height, int lenOutBoxes, int* outBoxes) {
    memset(&max_coord, 0, sizeof(int)*MAX_COMPONENTS*2);
    memset(&min_coord, 1, sizeof(int)*MAX_COMPONENTS*2);

    int max_idx = 0;
    for (int i=0; i<height; i++) {
        int row = i*width;
        for (int j=0; j<width; j++) {
            int component_id = markers[row+j]*2;
            if (markers[row+j] > max_idx) {
                max_idx = markers[row+j];
            }
            if (max_idx > MAX_COMPONENTS) {
                // THIS IS BAAAAD!!! Stoping to avoid stack corruption
                goto END_FOR;
            }
            if (j < min_coord[component_id]) {
                min_coord[component_id] = j;
            }
            if (j > max_coord[component_id]) {
                max_coord[component_id] = j;
            }
            if (i < min_coord[component_id+1]) {
                min_coord[component_id+1] = i;
            }
            if (i > max_coord[component_id+1]) {
                max_coord[component_id+1] = i;
            }
        }
    }
END_FOR:


    for (int i=0; i<=max_idx; i++) {
        if ((i*4+3) > lenOutBoxes) {
            break;
        }
        outBoxes[i*4+0] = min_coord[i*2];
        outBoxes[i*4+1] = min_coord[i*2+1];
        outBoxes[i*4+2] = max_coord[i*2];
        outBoxes[i*4+3] = max_coord[i*2+1];
    }
}
