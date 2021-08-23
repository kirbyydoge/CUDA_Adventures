#include<stdlib.h>

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define RGB_COMPONENT_COLOR 255

PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	//open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//read image format
	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	//check the image format
	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//alloc memory form image
	img = (PPMImage *)malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n');
		c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2)	{
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n');
	//memory allocation for pixel data
	img->data = (PPMPixel *) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//read pixel data from file
	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

void writePPM(const char *filename, PPMImage *img) {
	FILE *fp;
	//open file for output
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fp, "P6\n");

	//image size
	fprintf(fp, "%d %d\n", img->x, img->y);

	// rgb component depth
	fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

	// pixel data
	fwrite(img->data, 3 * img->x, img->y, fp);
	fclose(fp);
}

void writeGrayScale(const char *filename, unsigned char *img, int rows, int cols) {
	FILE *fp;
	//open file for output
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fp, "P2\n");

	//image size
	fprintf(fp, "%d %d\n", rows, cols);

	// rgb component depth
	fprintf(fp, "%d\n", 255);

	// pixel data
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			int idx = i * cols + j;
			fprintf(fp, "%d ", img[idx]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

uchar4 *PPM_to_uchar4(PPMImage *img, int alpha) {
	uchar4 *pixels = (uchar4 *) malloc(img->x * img->y * sizeof(uchar4));
	for(int i = 0; i < img->x; i++) {
		for(int j = 0; j < img->y; j++) {
			int idx = i * img->y + j;
			pixels[idx].x = img->data[idx].red;
			pixels[idx].y = img->data[idx].blue;
			pixels[idx].z = img->data[idx].green;
			pixels[idx].w = alpha;
		}
	}
	return pixels;
}

PPMImage *uchar4_to_PPM(uchar4 *img, int rows, int cols) {
	PPMImage *image =  (PPMImage *) malloc(sizeof(PPMImage));
	image->x = rows;
	image->y = cols;
	image->data = (PPMPixel *) malloc(rows * cols * sizeof(PPMPixel));
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			int idx = i * cols + j;
			image->data[idx].red = img[idx].x;
			image->data[idx].blue = img[idx].y;
			image->data[idx].green = img[idx].z;
		}
	}
	return image;
}