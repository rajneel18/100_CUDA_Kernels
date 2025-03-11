#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define WIDTH 1024
#define HEIGHT 768
#define SPHERE_RADIUS 0.5f
#define SPHERE_CENTER_X 0.0f
#define SPHERE_CENTER_Y 0.0f
#define SPHERE_CENTER_Z -1.5f

typedef struct {
    float x, y, z;
} Vec3;

__device__ Vec3 vec3(float x, float y, float z) {
    Vec3 v = {x, y, z};
    return v;
}

__device__ Vec3 vec3_add(Vec3 a, Vec3 b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ Vec3 vec3_mul(Vec3 v, float s) {
    return vec3(v.x * s, v.y * s, v.z * s);
}

__device__ float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3 vec3_normalize(Vec3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return vec3(v.x / len, v.y / len, v.z / len);
}

__global__ void render(unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = (y * WIDTH + x) * 3;
    float u = (2.0f * x / WIDTH - 1.0f);
    float v = (2.0f * y / HEIGHT - 1.0f);

    Vec3 ray_origin = vec3(0.0f, 0.0f, 0.0f);
    Vec3 ray_dir = vec3(u, v, -1.0f);
    ray_dir = vec3_normalize(ray_dir);

    Vec3 sphere_center = vec3(SPHERE_CENTER_X, SPHERE_CENTER_Y, SPHERE_CENTER_Z);
    Vec3 oc = vec3_sub(ray_origin, sphere_center);
    float a = vec3_dot(ray_dir, ray_dir);
    float b = 2.0f * vec3_dot(oc, ray_dir);
    float c = vec3_dot(oc, oc) - SPHERE_RADIUS * SPHERE_RADIUS;
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
        float t = (-b - sqrtf(discriminant)) / (2.0f * a);
        Vec3 hit_point = vec3_add(ray_origin, vec3_mul(ray_dir, t));
        Vec3 normal = vec3_normalize(vec3_sub(hit_point, sphere_center));

        Vec3 light_dir = vec3(1.0f, 1.0f, -1.0f);
        light_dir = vec3_normalize(light_dir);
        float intensity = fmaxf(0.0f, vec3_dot(normal, light_dir));

        image[idx] = (unsigned char)(255 * intensity);
        image[idx + 1] = (unsigned char)(50 * intensity);
        image[idx + 2] = (unsigned char)(50 * intensity);
    } else {
        image[idx] = 0;
        image[idx + 1] = 0;
        image[idx + 2] = 0;
    }
}

void save_image(unsigned char *image) {
    FILE *f = fopen("output.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image, 1, WIDTH * HEIGHT * 3, f);
    fclose(f);
}

int main() {
    unsigned char *d_image, *h_image;
    size_t image_size = WIDTH * HEIGHT * 3;
    cudaMalloc((void **)&d_image, image_size);
    h_image = (unsigned char *)malloc(image_size);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    render<<<numBlocks, threadsPerBlock>>>(d_image);
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    save_image(h_image);

    cudaFree(d_image);
    free(h_image);

    printf("Image saved as output.ppm\n");
    return 0;
}
