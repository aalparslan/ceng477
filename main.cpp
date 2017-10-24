#include "parser.h"
#include "ppm.h"
#include "scene_renderer.h"
#include <iostream>
#include <ctime>
using namespace parser;

int main(int argc, char *argv[]) {
  time_t begin = time(NULL);
  SceneRenderer scene_renderer(argv[1]);

  for (const Camera &camera : scene_renderer.Cameras()) {
    const Vec3i *pixels = scene_renderer.RenderImage(camera);
    const int width = camera.image_width;
    const int height = camera.image_height;
    unsigned char *image = new unsigned char[width * height * 3];

    int idx = 0;
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        const Vec3i pixel = pixels[i * height + j];
        image[idx++] = pixel.x;
        image[idx++] = pixel.y;
        image[idx++] = pixel.z;
      }
    }
    delete[] pixels;
    write_ppm(camera.image_name.c_str(), image, width, height);
    delete[] image;
  }
  time_t end = time(NULL);
  int elapsed_seconds = difftime(end,begin);
  std::cout << elapsed_seconds/60 << " minutes ";
  std::cout << elapsed_seconds%60 << " seconds" << std::endl; 
  return 0;
}
