#include <utils.h>
#include <cuda_runtime.h>

__global__ void ndvi(double* reflectance_band3, double* reflectance_band4, double* ndvi, int width_band) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;

    while (col < width_band) {

        ndvi[col] = (reflectance_band4[col] - reflectance_band3[col]) /
                         (reflectance_band4[col] + reflectance_band3[col]);

        col += blockIdx.x * blockDim.x;

    }

}

void read_line_tiff(TIFF* tif, double tif_line[], int line){
    if(TIFFReadScanline(tif, tif_line, line) < 0){
        cerr << "Read problem" << endl;
        exit(3);
    }
};

void write_line_tiff(TIFF* tif, double tif_line[], int line){

    if (TIFFWriteScanline(tif, tif_line, line) < 0){
        cerr << "Write problem!" << endl;
        exit(4);
    }

};

int main() {

    TIFF *reflectance_band3_tif, *reflectance_band4_tif, *ndvi_tif;

    reflectance_band3 = TIFFOpen("rfdf", "rm");

    reflectance_band4 = TIFFOpen("sdjhbsd", "rm");

    ndvi_tif = TIFFOpen("ndvi_path", "w8m");
    setup(ndvi_tif, reflectance_band3_tif);

    uint32 heigth_band, width_band;
    TIFFGetField(ndvi_tif, TIFFTAG_IMAGEWIDTH, &width_band);
    TIFFGetField(ndvi_tif, TIFFTAG_IMAGELENGTH, &heigth_band);

    double reflectance_band3[width_band], reflectance_band4[width_band], ndvi[width_band];
    double *dev_rb3, *dev_rb4, *dev_ndvi;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_rb3, width_band * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_rb4, width_band * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_ndvi, width_band * sizeof(double) ) );

    int threadNum = 256;
    int blockNum = (width_band + 256) / 256;

    for (int i = 0; i < heigth_band; i++) {

        read_line_tiff(reflectance_band3_tif, reflectance_band3, i);
        read_line_tiff(reflectance_band4_tif, reflectance_band4, i);

        HANDLE_ERROR( cudaMemcpy( dev_rb3,
            reflectance_band3,
            width_band * sizeof(double),
            cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMemcpy( dev_rb4,
                    reflectance_band4,
                    width_band * sizeof(double),
                    cudaMemcpyHostToDevice ) );

        ndvi<<blockNum, threadNum>>(dev_rb3, dev_rb4, dev_ndvi);

        HANDLE_ERROR( cudaMemcpy( ndvi,
                      dev_ndvi,
                      width_band * sizeof(double),
                      cudaMemcpyDeviceToHost ) );

        write_line_tiff(ndvi_tif, ndvi, i);

    }

    cudaFree(dev_rb3);
    cudaFree(dev_rb4);
    cudaFree(dev_ndvi);

    TIFFClose(reflectance_band3_tif);
    TIFFClose(reflectance_band4_tif);
    TIFFClose(ndvi_tif);

}