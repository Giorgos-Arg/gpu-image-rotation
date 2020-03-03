#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }
#define CHANNEL 3

// Struct for measuring performance
struct GpuTimer{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer(){
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer(){
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start(){
		cudaEventRecord(start, 0);
	}

	void Stop(){
		cudaEventRecord(stop, 0);
	}

	float Elapsed(){
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

struct Image {
	int width;
	int height;
	unsigned int bytes;
	unsigned char *img;
	unsigned char *dev_img;
};

// Reads a color ppm image file and saves the data in the provided Image structure. 
// The max_col_val is set to the value read from the input file.
// This is used later for writing the output image. 
int readInpImg(const char * fname, Image & source, int & max_col_val) {
	FILE *src;
	if (!(src = fopen(fname, "rb"))){
		printf("Couldn't open file %s for reading.\n", fname);
		return 1;
	}
	char p, s;
	fscanf(src, "%c%c\n", &p, &s);
	if (p != 'P' || s != '6'){	// Is it a valid format?
		printf("Not a valid PPM file (%c %c)\n", p, s);
		exit(1);
	}
	fscanf(src, "%d %d\n", &source.width, &source.height);
	fscanf(src, "%d\n", &max_col_val);
	int pixels = source.width * source.height;
	source.bytes = pixels * CHANNEL;  // CHANNEL = 3 => colored image with r, g, and b channels 
	source.img = (unsigned char *)malloc(source.bytes);
	if (fread(source.img, sizeof(unsigned char), source.bytes, src) != source.bytes){
		printf("Error reading file.\n");
		exit(1);
	}
	fclose(src);
	return 0;
}

// Write a color ppm image into a file.  
// Image structure represents the image in the memory. 
int writeOutImg(const char * fname, const Image & rotated, const int max_col_val) {
	FILE *out;
	if (!(out = fopen(fname, "wb"))){
		printf("Couldn't open file for output.\n");
		return 1;
	}
	fprintf(out, "P6\n%d %d\n%d\n", rotated.width, rotated.height, max_col_val);
	if (fwrite(rotated.dev_img, sizeof(unsigned char), rotated.bytes, out) != rotated.bytes){
		printf("Error writing file.\n");
		return 1;
	}
	fclose(out);
	return 0;
}

// To be launched on CPU
void rotate_90_CPU(unsigned char in[], unsigned char out[], int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int index_in = i * width * CHANNEL + j * CHANNEL;
			int index_out = j * height * CHANNEL + height * CHANNEL - (i + 1) * CHANNEL;
			out[index_out] = in[index_in];
			out[index_out + 1] = in[index_in + 1];
			out[index_out + 2] = in[index_in + 2];
		}
	}
}

// To be launched on a single thread
__global__ void rotate_90_serial(unsigned char in[], unsigned char out[], int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int index_in = i * width * CHANNEL + j * CHANNEL;
			int index_out = j * height * CHANNEL + height * CHANNEL - (i + 1) * CHANNEL;
			out[index_out] = in[index_in];
			out[index_out + 1] = in[index_in + 1];
			out[index_out + 2] = in[index_in + 2];
		}
	}
}

// To be launched with one thread per element
__global__ void rotate_90_parallel_per_element(unsigned char in[], unsigned char out[], int height, int width) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int index_in = i * width * CHANNEL + j * CHANNEL;
	int index_out = j * height * CHANNEL + height * CHANNEL - (i + 1) * CHANNEL;
	int image_size = height * width * CHANNEL;
	if(index_in < image_size && index_in >= 0 && index_out < image_size && index_out >= 0){
		out[index_out] = in[index_in];
		out[index_out + 1] = in[index_in + 1];
		out[index_out + 2] = in[index_in + 2];
	}
}

// To be launched with one thread per element. Thread blocks read & write tiles in coalesced fashion.
__global__ void rotate_90_parallel_per_element_tiled(unsigned char in[], unsigned char out[], int height, int width, unsigned int tile_size) {
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = blockIdx.x * tile_size, in_corner_j = blockIdx.y * tile_size * CHANNEL;
	int out_corner_i = blockIdx.y * tile_size * CHANNEL, out_corner_j = blockIdx.x * tile_size;
	int x = threadIdx.x, y = threadIdx.y;
	int index_tile = x * (tile_size)*CHANNEL + y * CHANNEL;
	int index_in = in_corner_i * width * CHANNEL + x * width *CHANNEL + in_corner_j + y * CHANNEL;
	int image_size = height * width * CHANNEL;
	int total_tile_size = tile_size * tile_size * CHANNEL;
	extern __shared__ unsigned char tile[];
	// Coalesced read from global mem, rotated write into shared mem:
	if(index_tile < total_tile_size && index_tile >= 0 && index_in < image_size  && index_in >= 0){
		tile[index_tile] = in[index_in];
		tile[index_tile+1] = in[index_in+1];
		tile[index_tile+2] = in[index_in+2];
	}

	__syncthreads();
	int index_out = out_corner_i*height + y * height * CHANNEL + height * CHANNEL - (x+1) * CHANNEL - out_corner_j * CHANNEL;
	// Read from shared mem, coalesced write to global mem:
	if(index_tile < total_tile_size && index_tile >= 0 && index_out < image_size && index_out >= 0 && index_in < image_size  && index_in >= 0){
		out[index_out] = tile[index_tile];
		out[index_out + 1] = tile[index_tile + 1];
		out[index_out + 2] = tile[index_tile + 2];
	}

}

// To be launched with one thread per element. Thread blocks read & write tiles in coalesced fashion.
// Shared memory array padded to avoid bank conflicts.
__global__ void rotate_90_parallel_per_element_tiled_padded(unsigned char in[], unsigned char out[], int height, int width, unsigned int tile_size) {
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = blockIdx.x * tile_size, in_corner_j = blockIdx.y * tile_size * CHANNEL;
	int out_corner_i = blockIdx.y * tile_size * CHANNEL, out_corner_j = blockIdx.x * tile_size;
	int x = threadIdx.x, y = threadIdx.y;
	int image_size = height * width * CHANNEL;
	int total_tile_size = tile_size * (tile_size + 1) * CHANNEL ;
	int index_tile = x * (tile_size + 1)*CHANNEL + y * CHANNEL;
	int index_in = in_corner_i * width * CHANNEL + x * width *CHANNEL + in_corner_j + y * CHANNEL;
	extern __shared__ unsigned char tile[];
	// coalesced read from global mem, rotated write into shared mem:
	if(index_tile < total_tile_size && index_tile >= 0 && index_in < image_size){
		tile[index_tile] = in[index_in];
		tile[index_tile+1] = in[index_in+1];
		tile[index_tile+2] = in[index_in+2];
	}
	__syncthreads();
	int index_out = out_corner_i*height + y * height * CHANNEL + height * CHANNEL - (x+1)*CHANNEL - out_corner_j*CHANNEL;
	// Read from shared mem, coalesced write to global mem:
	if(index_tile < total_tile_size && index_tile >= 0 && index_out < image_size && index_out >= 0 && index_in < image_size && index_in >=0 ){
		out[index_out] = tile[index_tile];
		out[index_out + 1] = tile[index_tile + 1];
		out[index_out + 2] = tile[index_tile + 2];
	}
}


int main(int argc, char **argv){
	if (argc != 3){
		printf("Usage: exec filename kernel\n");
		exit(1);
	}
	char *fname = argv[1];
	char kname[100] = ""; // kernel name
	int choice = atoi(argv[2]); // kernel choice
	Image source;
	int max_col_val;
	GpuTimer timer;
	unsigned char *d_in, *d_out;
	// Read the input file
	if (readInpImg(fname, source, max_col_val) != 0)  
		exit(1);
	source.dev_img = (unsigned char *)malloc(source.bytes);
	CHECK(cudaMalloc(&d_in, source.bytes));
	CHECK(cudaMalloc(&d_out, source.bytes));
	CHECK(cudaMemcpy(d_in, source.img, source.bytes, cudaMemcpyHostToDevice));
	// Run selected kernel
	switch (choice) {
	case 1: // Serial execution on GPU, i.e. creating ONLY ONE thread
		strcpy(kname, "Serial execution on GPU");
		timer.Start();
		rotate_90_serial <<<1, 1>>> (d_in, d_out, source.height, source.width);
		timer.Stop();
		CHECK(cudaMemcpy(source.dev_img, d_out, source.bytes, cudaMemcpyDeviceToHost));
		break;
	case 2: { // One thread per pixel
		strcpy(kname, "One thread per pixel");
		int k = 32;
		dim3 blocks(ceil((float)source.height / (float)k), ceil((float)source.width / (float)k));   // blocks per grid (using ceil in case height or width are not multiple of k)
		dim3 threads(k, k);	// threads per block
		timer.Start();
		rotate_90_parallel_per_element <<<blocks, threads>>> (d_in, d_out, source.height, source.width);
		timer.Stop();
		CHECK(cudaMemcpy(source.dev_img, d_out, source.bytes, cudaMemcpyDeviceToHost));
		break;
	}
	case 3: { // One thread per pixel - tiled (16 X 16)
		strcpy(kname, "One thread per pixel - tiled (16 X 16)");
		int k = 16; // tile size is k x k
		unsigned int shmem_size = k * k * CHANNEL * sizeof(unsigned char);
		dim3 blocks(ceil((float)source.height / (float)k), ceil((float)source.width / (float)k));   // blocks per grid 
		dim3 threads(k, k);	// threads per block
		timer.Start();
		rotate_90_parallel_per_element_tiled <<<blocks, threads, shmem_size>>> (d_in, d_out, source.height, source.width, k);
		timer.Stop();
		CHECK(cudaMemcpy(source.dev_img, d_out, source.bytes, cudaMemcpyDeviceToHost));
		break;
	}
	case 4: { // One thread per matrix element - tiled (16x16) - no shared mem conflict
		strcpy(kname, "One thread per matrix element - tiled (16x16) - no shared mem conflict");
		int k = 16; // tile size is k x k
		dim3 blocks(ceil((float)source.height / (float)k), ceil((float)source.width / (float)k));   // blocks per grid
		dim3 threads(k, k);	// threads per block
		unsigned int shmem_size = k * (k+1) * CHANNEL * sizeof(unsigned char);
		timer.Start();
		rotate_90_parallel_per_element_tiled_padded <<<blocks, threads, shmem_size>>> (d_in, d_out, source.height, source.width, k);
		timer.Stop();
		CHECK(cudaMemcpy(source.dev_img, d_out, source.bytes, cudaMemcpyDeviceToHost));
		break;
	}
	default:
		printf("Choose a kernel between 1 and 4");
		exit(1);
	}
	printf("\nRotating Image \"%s\" with Height = %d and Width = %d.\nUsing kernel %d: %s\nElapsed time: %g ms.\n\n", argv[1],
		source.height, source.width, choice, kname, timer.Elapsed());
	// Swap height and width for the rotated image
	int temp = source.height;
	source.height = source.width;
	source.width = temp;
	// Write the output file
	if (writeOutImg("rotated.ppm", source, max_col_val) != 0) // For demonstration, the input file is written to a new file named "rotated.ppm" 
		exit(1);
	// free up the allocated memory
	free(source.img);
	free(source.dev_img);
	CHECK(cudaFree(d_in));
	CHECK(cudaFree(d_out));
	exit(0);
}