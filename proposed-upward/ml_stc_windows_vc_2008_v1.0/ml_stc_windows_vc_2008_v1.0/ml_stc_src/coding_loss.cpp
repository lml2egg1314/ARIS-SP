#include <iostream>
#include <iomanip>
#include <cmath>
#include "coding_loss.h"

// binary entropy function in double precission
inline double binary_entropy(double x) {

    double const LOG2 = log(2.0);
    double const EPS = std::numeric_limits<double>::epsilon();
    double z;

    if ((x<EPS) || ((1-x)<EPS)) {
        return 0;
    } else {
        z = (-x*log(x)-(1-x)*log(1-x))/LOG2;
        return z;
    }
}

double calc_coding_loss(const u8 *cover, unsigned int cover_length, unsigned int message_length, const double *costs, const u8 *stego) {
    
    unsigned int i;
    double avg_distortion = 0; // average distortion between cover and stego
    double opt_rel_payload = 0;
    double beta, actual_rel_payload, coding_loss;

    // calculate average distortion per cover element
    for(i=0;i<cover_length;i++) { avg_distortion += (cover[i]==stego[i]) ? 0 : costs[i]; }
    avg_distortion /= cover_length;

    // find optimal value of parameter beta such that optimal coding achieves avg_distortion
    beta = calculate_beta_from_distortion(costs, cover_length, avg_distortion);
    // calculate optimal relative payload achieved by optimal coding method
    for(i=0;i<cover_length;i++) {
        double x = exp(-beta*costs[i]);
        opt_rel_payload += binary_entropy(x/(1.0+x)); // measure entropy of the optimal flipping noise
    }
    opt_rel_payload /= cover_length;
    actual_rel_payload = ((double)message_length)/((double)cover_length);

    coding_loss = 1.0 - actual_rel_payload/opt_rel_payload;
    return coding_loss;
}

/*
  find beta such that average distortion caused by optimal flipping noise is avg_distortion
  using binary search
*/
double calculate_beta_from_distortion(const double *costs, unsigned int n, double avg_distortion) {

    double dist1, dist2, dist3, beta1, beta2, beta3;
    int j = 0;
    unsigned int i;
    const double INF = std::numeric_limits<double>::infinity();

    beta2 = -1; // initial value - error
    beta1 = 0; dist1 = 0;
    for(i=0;i<n;i++) {
        dist1 += (costs[i]==INF) ? 0 : costs[i];
    }
    dist1 /= n*2;
    beta3 = 1e+5; dist3 = avg_distortion+1; // this is just an initial value
    while(dist3>avg_distortion) {
        beta3 *= 2;
        for(i=0;i<n;i++) {
            double p_flip = 1.0-1.0/(1.0+exp(-beta3*costs[i]));
            if (costs[i]!=INF) { dist3 += costs[i]*p_flip; }
        }
        dist3 /= n;
        j++;
        if (j>100) {
            // beta is probably unbounded => it seems that we cannot find beta such that
            // relative distortion will be smaller than requested. Binary search does not make sense here.
            return -1;
        }
    }
    while ( (dist1-dist3>(avg_distortion/1000.0)) ) { // iterative search for parameter beta
        beta2 = beta1+(beta3-beta1)/2; dist2 = 0;
        for(i=0;i<n;i++) {
            if (costs[i]!=INF) {
                double p_flip = 1.0-1.0/(1.0+exp(-beta2*costs[i]));
                if (costs[i]!=INF) { dist2 += costs[i]*p_flip; }
            }
        }
        dist2 /= n;
        if (dist2<avg_distortion) {
            beta3 = beta2;
            dist3 = dist2;
        } else {
            beta1 = beta2;
            dist1 = dist2;
        }
    }

    return beta2;
}
