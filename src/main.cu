#include "utils.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void hoCalc(double* rn, double* soilHeat,
		double* ho, int width_band) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;

	while (col < width_band) {

		ho[col] = rn[col] - soilHeat[col];

		col += blockDim.x * gridDim.x;

	}

}

void read_line_tiff(TIFF* tif, double tif_line[], int line) {
	if (TIFFReadScanline(tif, tif_line, line) < 0) {
		std::cerr << "Read problem" << std::endl;
		exit(3);
	}
}
;

void write_line_tiff(TIFF* tif, double tif_line[], int line) {

	if (TIFFWriteScanline(tif, tif_line, line) < 0) {
		std::cerr << "Write problem!" << std::endl;
		exit(4);
	}

}
;

void setup(TIFF* new_tif, TIFF* base_tif) {
	uint32 image_width, image_length;

	TIFFGetField(base_tif, TIFFTAG_IMAGEWIDTH, &image_width);
	TIFFGetField(base_tif, TIFFTAG_IMAGELENGTH, &image_length);

	TIFFSetField(new_tif, TIFFTAG_IMAGEWIDTH, image_width);
	TIFFSetField(new_tif, TIFFTAG_IMAGELENGTH, image_length);
	TIFFSetField(new_tif, TIFFTAG_BITSPERSAMPLE, 64);
	TIFFSetField(new_tif, TIFFTAG_SAMPLEFORMAT, 3);
	TIFFSetField(new_tif, TIFFTAG_COMPRESSION, 1);
	TIFFSetField(new_tif, TIFFTAG_PHOTOMETRIC, 1);
	TIFFSetField(new_tif, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(new_tif, TIFFTAG_ROWSPERSTRIP, 1);
	TIFFSetField(new_tif, TIFFTAG_RESOLUTIONUNIT, 1);
	TIFFSetField(new_tif, TIFFTAG_XRESOLUTION, 1);
	TIFFSetField(new_tif, TIFFTAG_YRESOLUTION, 1);
	TIFFSetField(new_tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
}
;

int main() {

	cudaProfilerStart();

	TIFF *rn_tif, *soilHeat_tif, *ho_tif;

	rn_tif = TIFFOpen("/home/itallo/Downloads/Rn.tif", "rm");

	soilHeat_tif = TIFFOpen("/home/itallo/Downloads/G.tif", "rm");

	ho_tif = TIFFOpen("/home/itallo/Downloads/out.tif", "w8m");
	setup(ho_tif, rn_tif);

	uint32 heigth_band, width_band;
	TIFFGetField(ho_tif, TIFFTAG_IMAGEWIDTH, &width_band);
	TIFFGetField(ho_tif, TIFFTAG_IMAGELENGTH, &heigth_band);

	double rn[width_band], soilHeat[width_band],
			ho[width_band];
	double *dev_rn, *dev_soilHeat, *dev_ho;

	HANDLE_ERROR(cudaMalloc((void** )&dev_rn, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void** )&dev_soilHeat, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void** )&dev_ho, width_band * sizeof(double)));

	int threadNum = 256;
	int blockNum = (width_band + 256) / 256;

	for (int i = 0; i < heigth_band; i++) {

		read_line_tiff(rn_tif, rn, i);
		read_line_tiff(soilHeat_tif, soilHeat, i);

		HANDLE_ERROR(
				cudaMemcpy(dev_rn, rn,
						width_band * sizeof(double), cudaMemcpyHostToDevice));

		HANDLE_ERROR(
				cudaMemcpy(dev_soilHeat, soilHeat,
						width_band * sizeof(double), cudaMemcpyHostToDevice));

		hoCalc<<<blockNum, threadNum>>>(dev_rn, dev_soilHeat, dev_ho,
				width_band);

		HANDLE_ERROR(
				cudaMemcpy(ho, dev_ho, width_band * sizeof(double),
						cudaMemcpyDeviceToHost));

		write_line_tiff(ho_tif, ho, i);

	}

	cudaFree(dev_rn);
	cudaFree(dev_soilHeat);
	cudaFree(dev_ho);

	cudaProfilerStop();
	cudaDeviceReset();

	TIFFClose(rn_tif);
	TIFFClose(soilHeat_tif);
	TIFFClose(ho_tif);

}
