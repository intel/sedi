#ifndef CMD_LINE_PARSER_H
#define CMD_LINE_PARSER_H

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

class CmdLineParser {
	public:

	long ng1, ng2, ng3;
	int  np1, np2, np3;

	int samples;
	double tolerance;
	int maxiter;

	int solver_mode;
	bool ignore_solver_errors;
	bool dont_enable_fp_exceptions;

	char *progname;
	char *validate_solver_h5_file;
	char *validate_estimator_h5_file;

	CmdLineParser(int argc, char * argv[]) {

		char *ng_buf;
		char *np_buf;
		char *samples_buf;
		char *tol_buf;
		char *maxiter_buf;
		char *bcg_mode_buf;

		ignore_solver_errors = false;
		dont_enable_fp_exceptions = false;

		progname = argv[0];
		ng_buf = np_buf = samples_buf = tol_buf = maxiter_buf = bcg_mode_buf = 00;
		validate_solver_h5_file = 00;
		validate_estimator_h5_file = 00;
		int c;
		while ((c = getopt (argc, argv, "g:p:s:e:m:if:FV:E:")) != -1) {
			switch (c)
			{
				case 'g':
					ng_buf = optarg;
					break;
				case 'p':
					np_buf = optarg;
					break;
				case 's':
					samples_buf = optarg;
					break;
				case 'e':
					tol_buf = optarg;
					break;
				case 'm':
					maxiter_buf = optarg;
					break;
				case 'i':
					ignore_solver_errors = true;
					break;
				case 'f':
					bcg_mode_buf = optarg;
					break;
				case 'F':
					dont_enable_fp_exceptions = true;
					break;
				case 'V':
					validate_solver_h5_file = optarg;
					break;
				case 'E':
					validate_estimator_h5_file = optarg;
					break;
				default:
					usage();
			}
		}

		if (ng_buf != 00) {
			int nread = sscanf(ng_buf, "%ldx%ldx%ld", &ng1, &ng2, &ng3);
			if (nread != 3) usage();
		} else {
			usage();
		}

		if (np_buf != 00) {
			int nread = sscanf(np_buf, "%dx%dx%d", &np1, &np2, &np3);
			if (nread != 3) usage();
		} else {
			usage();
		}

		if (samples_buf != 00) {
			int nread = sscanf(samples_buf, "%d", &samples);
			if (nread != 1) usage();
		} else {
			samples = 400;
		}

		if (tol_buf != 00) {
			int nread = sscanf(tol_buf, "%lf", &tolerance);
			if (nread != 1) usage();
		} else {
			tolerance = 1e-4;
		}

		if (maxiter_buf != 00) {
			int nread = sscanf(maxiter_buf, "%d", &maxiter);
			if (nread != 1) usage();
		} else {
			maxiter = 1000;
		}

		if (bcg_mode_buf != 00) {
			int nread = sscanf(bcg_mode_buf, "%d", &solver_mode);
			if (nread != 1) usage();
		} else {
			solver_mode = 0; // Default behavior for solver
		}
	}

	private:
	void usage() {
		fprintf(stderr, "Usage: %s -g NxNxN -p NxNxN [-t NxNxN] [-s NSAMPLES] [-e TOL] [-m MAXITER] [-S]\n", progname);
		fprintf(stderr, "\t-g NxNxN : size of the global 3D grid\n");
		fprintf(stderr, "\t-p NxNxN : decomposition into ranks\n");
		fprintf(stderr, "\t-s NSAMPLES : number of samples for stochastic estimation\n");
		fprintf(stderr, "\t-e TOL : solver tolerance\n");
		fprintf(stderr, "\t-m MAXITER : solver maximum iterations\n");
		fprintf(stderr, "\t-i : ignore solver convergence errors and keep going\n");
		fprintf(stderr, "\t-f : [advanced] flag for solver operation mode, see e.g. BiCgStab::mode_t\n");
		fprintf(stderr, "\t-F : don't enable floating point exceptions (keep going if FPE occurs) [Linux only]\n");
		fprintf(stderr, "\t-V HDF5_FILE : enable solver validation mode against HDF5 archive [requires HDF5]\n");
		fprintf(stderr, "\t-E HDF5_FILE : enable estimator validation mode against HDF5 archive [requires HDF5]\n");
		exit(1);
	}
};

#endif
