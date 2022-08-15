#include "stc_ml_c.h"
#include "mex.h"

// input and output arguments
#define OUT_DISTORTION   plhs[0] // extracted message represented as uint8 array
#define OUT_STEGO        plhs[1] // 
#define OUT_NUM_MSG_BITS plhs[2] // array describing the number of bits embedded in every layer
#define OUT_CODING_LOSS  plhs[3] // OPTIONAL - describes the loss introduced by practical algorithm

#define IN_COVER         prhs[0] // array of cover elements
#define IN_COSTS         prhs[1] // array of costs of changing each element by +-1 or 0
#define IN_MESSAGE       prhs[2] // array with message bits in uint8
#define IN_CONSTRAINT_H  prhs[3] // constraint height of STC codes, default = 10
#define IN_WET_COST      prhs[4] // cost used to represent forbidden stego element, default = Inf

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	uint h, n, m, trials = 10;
    int *stego, *cover;
    float *costs, *coding_loss;
    u8* message;

	// check for proper number of input and output arguments
	if (nrhs<3) { mexErrMsgTxt("At least three input arguments are required."); } 

	if (mxGetM(IN_COVER)>1 || mxGetN(IN_COVER)==1) mexErrMsgTxt("Vector of cover elements must be a row vector.");
    if (!mxIsClass(IN_COVER, "int32")) mexErrMsgTxt("Vector of cover elements must be of type int32.");
	cover = (int*)mxGetPr(IN_COVER);
	n = mxGetN(IN_COVER);

    if (mxGetM(IN_COSTS)!=5 || mxGetN(IN_COSTS)!=mxGetN(IN_COVER)) mexErrMsgTxt("Matrix of costs must be 5xn, where n is number of cover elements.");
    if (!mxIsClass(IN_COSTS, "single")) mexErrMsgTxt("Matrix of costs must be of type single.");
	costs = (float*)mxGetPr(IN_COSTS);

    if (mxGetM(IN_MESSAGE)!=1) mexErrMsgTxt("Message vector must be a row vector.");
    if (!mxIsClass(IN_MESSAGE, "uint8")) mexErrMsgTxt("Message array must be of type uint8.");
	message = (u8*)mxGetPr(IN_MESSAGE);
    m = mxGetN(IN_MESSAGE);

	// get constraint length
    h = 10; // default value
    if (nrhs>3) {
        if (mxGetM(IN_CONSTRAINT_H)>1 || mxGetN(IN_CONSTRAINT_H)>1) mexErrMsgTxt("Constraint height of the STC must be scalar integer value.");
        h = (uint)mxGetScalar(IN_CONSTRAINT_H);
    }
    // wet cost
    float wet_cost = F_INF; // default value is infinity
    if (nrhs>4) {
        if (mxGetM(IN_WET_COST)>1 || mxGetN(IN_WET_COST)>1) mexErrMsgTxt("Wet cost must be scalar value.");
        if (!mxIsClass(IN_WET_COST, "single")) mexErrMsgTxt("Wet cost must be of type single.");
        wet_cost = (float)mxGetScalar(IN_WET_COST);
    }

    coding_loss = 0; // this is optional output argument
    if (nlhs>3) {
        OUT_CODING_LOSS = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        coding_loss = (float*)mxGetPr(OUT_CODING_LOSS);
    }

    OUT_DISTORTION = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    double* dist = mxGetPr(OUT_DISTORTION);
    OUT_STEGO = mxCreateNumericMatrix(1, n, mxINT32_CLASS, mxREAL);
    stego = (int*)mxGetPr(OUT_STEGO);
    OUT_NUM_MSG_BITS = mxCreateNumericMatrix(1, 3, mxUINT32_CLASS, mxREAL);
    uint *num_msg_bits = (uint*)mxGetPr(OUT_NUM_MSG_BITS);

    *dist = stc_pm2_pls_embed(n, cover, costs, m, message, h, wet_cost, stego, num_msg_bits, trials, coding_loss);
}
