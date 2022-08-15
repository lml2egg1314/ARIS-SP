#include "stc_ml_c.h"
#include "mex.h"

// input and output arguments
#define OUT_MESSAGE     plhs[0] // extracted message represented as uint8 array
#define IN_STEGO        prhs[0] // array of stego elements
#define IN_NUM_MSG_BITS prhs[1] // array describing the number of bits embedded in every layer
#define IN_CONSTRAINT_H prhs[2] // constraint height of STC codes, default = 10

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	uint h, n;
    int* stego;

	// check for proper number of input and output arguments
	if (nrhs<2) { mexErrMsgTxt("At leat two input arguments are required."); } 

	if (mxGetM(IN_STEGO)>1 || mxGetN(IN_STEGO)==1) mexErrMsgTxt("Vector of stego elements must be a row vector.");
    if (!mxIsClass(IN_STEGO, "int32")) mexErrMsgTxt("Vector of stego elements must be of type int32.");
	stego = (int*)mxGetPr(IN_STEGO);
	n = mxGetN(IN_STEGO);

	if (mxGetM(IN_NUM_MSG_BITS)>1 || (!mxIsClass(IN_NUM_MSG_BITS, "uint32"))) mexErrMsgTxt("Vector of number of embedded bits must be uint32 row vector.");
    uint* num_msg_bits = (uint*)mxGetPr(IN_NUM_MSG_BITS);
    uint num_of_layers = mxGetN(IN_NUM_MSG_BITS);

	// get constraint length
    h = 10; // default value
    if (nrhs>2) {
        if (mxGetM(IN_CONSTRAINT_H)>1 || mxGetN(IN_CONSTRAINT_H)>1) mexErrMsgTxt("Constraint height of the STC must be scalar integer value.");
        h = (uint)mxGetScalar(IN_CONSTRAINT_H);
    }
    uint m = 0;
    for (uint i=0; i<num_of_layers; i++) m += num_msg_bits[i];
    OUT_MESSAGE = mxCreateNumericMatrix(1, m, mxUINT8_CLASS, mxREAL);
    u8* message = (u8*)mxGetPr(OUT_MESSAGE);

    stc_ml_extract(n, stego, num_of_layers, num_msg_bits, h, message);
}
