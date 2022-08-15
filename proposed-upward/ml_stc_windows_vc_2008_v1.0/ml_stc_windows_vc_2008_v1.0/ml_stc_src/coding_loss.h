#ifndef CODING_LOSS_H_
#define CODING_LOSS_H_

/*
 CODING LOSS: add description and definition of coding loss ...

 * **********************************************************************************
 * EXAMPLE:                                                                         *
 *    stc_embed(cover, n, msg, m, (void*)profile, true, stego, constraint_height);  *
 *    coding_loss = calc_coding_loss(cover, n, m, profile, stego);                  *
 *    stc_extract(stego, n, msg2, m, constraint_height);                            *
 * **********************************************************************************
*/

#include <cmath>
#include <limits>

typedef unsigned char u8;

double calc_coding_loss(const u8 *cover, unsigned int cover_length, unsigned int message_length, const double *costs, const u8 *stego);
double calculate_beta_from_distortion(const double *costs, unsigned int n, double avg_distortion);

#endif
