#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <inttypes.h>
#include "routinesGPU.h"

#define DEG2RAD 0.017453f

__global__ void noiseReduction(uint8_t *im, int height, int width, float *NR)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i < height - 2 && i >= 2) && (j < width - 2 && j >= 2))
	{
		NR[i*width+j] =
			 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
			+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
			+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
			+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
			+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
			/159.0;
	}
}

__global__ void calculateGradient(int height, int width, float *NR, float *G,
	float *phi, float *Gx, float *Gy, float PI)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i < height - 2 && i >= 2) && (j < width - 2 && j >= 2))
	{
		Gx[i*width+j] =
			 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
			+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
			+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
			+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


		Gy[i*width+j] =
			 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
			+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
			+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

		G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
		phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));


		if(fabs(phi[i*width+j])<=PI/8 )
			phi[i*width+j] = 0;
		else if (fabs(phi[i*width+j])<= 3*(PI/8))
			phi[i*width+j] = 45;
		else if (fabs(phi[i*width+j]) <= 5*(PI/8))
			phi[i*width+j] = 90;
		else if (fabs(phi[i*width+j]) <= 7*(PI/8))
			phi[i*width+j] = 135;
		else phi[i*width+j] = 0;
	}

}
__global__ void calculateEdges(int height, int width, uint8_t * pedge, float *phi, float *G)
{
	// Edge
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i < height - 3 && i >= 3) && (j < width - 3 && j >= 3))
	{
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
	}
}
__global__ void hystheresisThresholding(int height, int width, uint8_t * pedge, float *G, float level, uint8_t *image_out)
{

		int ii, jj;
	float lowthres, hithres;
	lowthres = level/2;
	hithres  = 2*(level);
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i < height - 3 && i >= 3) && (j < width - 3 && j >= 3))
	{
			image_out[i*width+j] = 0;
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
	}
}
void canny(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level,
	int height, int width, dim3 dimBlock, dim3 dimGrid)
	{
		float PI = 3.141593;

		noiseReduction<<<dimGrid,dimBlock>>>(im, height, width, NR);
		cudaThreadSynchronize();
		calculateGradient<<<dimGrid, dimBlock>>>(height, width, NR, G,
			phi, Gx, Gy, PI);
		cudaThreadSynchronize();
		calculateEdges<<<dimGrid, dimBlock>>>(height,width,pedge,phi,G);
		cudaThreadSynchronize();
		hystheresisThresholding<<<dimGrid, dimBlock>>>(height, width, pedge, G, level, image_out);
		cudaThreadSynchronize();
	}

	__global__ void houghtransformKernel(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height,
		float *sin_table, float *cos_table)
	{
		int i, j, theta;
		i = blockIdx.x * blockDim.x + threadIdx.x;
		j = blockIdx.y * blockDim.y + threadIdx.y;
		float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

		if(i < accu_width*accu_height)
			accumulators[i]=0;

		float center_x = width/2.0;
		float center_y = height/2.0;
		if(i < height)
		{
			if(j < width)
			{
				if( im[ (i*width) + j] > 250 ) // Pixel is edge
				{
					for(theta=0;theta<180;theta++)
					{
						float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
						//printf("%.6f \n", rho);
						atomicAdd(&accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta], 1);
						//accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta]++;
						//printf("%d\n", accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta]);
					}
				}
			}
		}
	}
	void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height,
		float *sin_table, float *cos_table, dim3 dimBlock, dim3 dimGrid)
	{
		houghtransformKernel<<<dimGrid,dimBlock>>>(im, width, height, accumulators, accu_width, accu_height, sin_table, cos_table);
		cudaThreadSynchronize();
	}

	void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height,
		float *sin_table, float *cos_table,
		int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
	{
		int rho, theta, ii, jj;
		uint32_t max;
		for(rho=0;rho<accu_height;rho++)
		{
			for(theta=0;theta<accu_width;theta++)
			{
				//printf("accumulator: %d \n", accumulators[(rho*accu_width) + theta]);
				if(accumulators[(rho*accu_width) + theta] >= threshold)
				{
					printf("%d\n", accumulators[(rho*accu_width) + theta]);
					//Is this point a local maxima (9x9)
					max = accumulators[(rho*accu_width) + theta];
					for(int ii=-4;ii<=4;ii++)
					{
						for(int jj=-4;jj<=4;jj++)
						{
							if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )
							{
								if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )
								{
									max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
								}
							}
						}
					}
					if(max == accumulators[(rho*accu_width) + theta]) //local maxima
					{
						int x1, y1, x2, y2;
						x1 = y1 = x2 = y2 = 0;

						if(theta >= 45 && theta <= 135)
						{
							if (theta>90) {
								//y = (r - x cos(t)) / sin(t)
								x1 = width/2;
								y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
								x2 = width;
								y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							} else {
								//y = (r - x cos(t)) / sin(t)
								x1 = 0;
								y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
								x2 = width*2/5;
								y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							}
						} else {
							//x = (r - y sin(t)) / cos(t);
							y1 = 0;
							x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);
							y2 = height;
							x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);
						}
						x1_lines[*lines] = x1;
						y1_lines[*lines] = y1;
						x2_lines[*lines] = x2;
						y2_lines[*lines] = y2;
						(*lines)++;
					}
				}
			}
		}
	}

	uint8_t *image_RGB2BW(uint8_t *image_in, int height, int width)
	{
		int i, j;
		uint8_t *imageBW = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
		float R, B, G;

		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++)
			{
				R = (float)(image_in[3 * (i * width + j)]);
				G = (float)(image_in[3 * (i * width + j) + 1]);
				B = (float)(image_in[3 * (i * width + j) + 2]);

				imageBW[i * width + j] = (uint8_t)(0.2989 * R + 0.5870 * G + 0.1140 * B);
			}

		return imageBW;
	}

	void draw_lines(uint8_t *imgtmp, int width, int height, int *x1, int *y1, int *x2, int *y2, int nlines)
	{
		int x, y, wl, l;
		int width_line=9;

		for(l=0; l<nlines; l++)
			for(wl=-(width_line>>1); wl<=(width_line>>1); wl++)
				for (x=x1[l]; x<x2[l]; x++)
				{
					y = (float)(y2[l]-y1[l])/(x2[l]-x1[l])*(x-x1[l])+y1[l]; //Line eq. known two points
					if (x+wl>0 && x+wl<width && y>0 && y<height)
					{
						imgtmp[3*((y)*width+x+wl)  ] = 255;
						imgtmp[3*((y)*width+x+wl)+1] = 0;
						imgtmp[3*((y)*width+x+wl)+2] = 0;
					}
				}
	}

	void line_asist_GPU(uint8_t *im, int height, int width,
		uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
		float *sin_table, float *cos_table,
		uint32_t *accum, int accu_height, int accu_width,
		int *x1, int *x2, int *y1, int *y2, int *nlines)
	{
		dim3 dimBlock(16,16,1);
		dim3 dimGrid(ceil(height/16.0), ceil(width/16.0), 1);

			int threshold;
			uint8_t* im_GPU;
			uint8_t* imEdge_GPU;
			uint8_t* pedge_GPU;
			float* NR_GPU;
			float* G_GPU;
			float* phi_GPU;
			float* Gx_GPU;
			float* Gy_GPU;
			float* sin_table_GPU;
			float* cos_table_GPU;
			uint32_t* accum_GPU;

			cudaMalloc(&NR_GPU, width * height * sizeof(float));
			cudaMalloc(&G_GPU, width * height * sizeof(float));
			cudaMalloc(&phi_GPU, width * height * sizeof(float));
			cudaMalloc(&Gx_GPU, width * height * sizeof(float));
			cudaMalloc(&Gy_GPU, width * height * sizeof(float));

			cudaMalloc(&sin_table_GPU, 180 * sizeof(float));
			cudaMemcpy(sin_table_GPU, sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);

			cudaMalloc(&cos_table_GPU, 180 * height * sizeof(float));
			cudaMemcpy(cos_table_GPU, cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);

			cudaMalloc(&accum_GPU, accu_width * accu_height * sizeof(uint32_t));

			cudaMalloc(&imEdge_GPU, width * height * sizeof(uint8_t));
			cudaMalloc(&pedge_GPU, width * height * sizeof(uint8_t));

			cudaMalloc(&im_GPU, width * height * sizeof(uint8_t));
			cudaMemcpy(im_GPU, im, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

		/* Canny */
		canny(im_GPU, imEdge_GPU,
			NR_GPU, G_GPU, phi_GPU, Gx_GPU, Gy_GPU, pedge_GPU,
			1000.0f, //level
			height, width, dimBlock, dimGrid);

		/* hough transform */
		houghtransform(imEdge_GPU, width, height, accum_GPU, accu_width, accu_height, sin_table_GPU, cos_table_GPU, dimBlock, dimGrid);
		cudaMemcpy(accum, accum_GPU, accu_width * accu_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		// for(int i = 0; i < accu_width * accu_height; i++)
		// 	if(accum[i] != 0)
		//  		printf("accumulator: %d \n", accum[i]);
		if (width>height) threshold = width/6;
		else threshold = height/6;
		printf("width: %d \n", width);
		printf("height: %d \n", height);
		printf("threshold: %d \n", threshold);


		getlines(threshold, accum, accu_width, accu_height, width, height,
			sin_table, cos_table,
			x1, y1, x2, y2, nlines);

		/* To do */
		cudaMemcpy(im, im_GPU,sizeof(uint8_t)*width * height, cudaMemcpyDeviceToHost);
		printf("líneas: %d\n",*nlines);
		free(imEdge);
		free (NR);
		free (G);
		free (phi);
		free(Gx);
		free(Gy);
		free(pedge);
		free(accum);
		// free(x1);
		// free(x2);
		// free(y1);
		// free(y2);
		//free(nlines);
		cudaFree(imEdge_GPU); cudaFree(im_GPU); cudaFree(pedge_GPU); cudaFree(NR_GPU); cudaFree(G_GPU); cudaFree(phi_GPU); cudaFree(Gx_GPU); cudaFree(Gy_GPU); cudaFree(accum_GPU);

	}
