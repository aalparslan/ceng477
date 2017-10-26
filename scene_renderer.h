#ifndef _SCENE_RENDERER_H
#define _SCENE_RENDERER_H

#include "parser.h"

class SceneRenderer {
private:
  parser::Vec3f q, usu, vsv;
  parser::Scene scene_;

  float DoesIntersect(const parser::Vec3f &origin,
                      const parser::Vec3f &distance, const parser::Face &face);
  float DoesIntersect(const parser::Vec3f &origin,
                      const parser::Vec3f &distance, const parser::Mesh &mesh,
                      parser::Face &intersecting_face, float tmin);
  float DoesIntersect(const parser::Vec3f &origin,
                      const parser::Vec3f &distance,
                      const parser::Triangle &triangle);
  float DoesIntersect(const parser::Vec3f &origin,
                      const parser::Vec3f &distance,
                      const parser::Sphere &sphere);

  parser::Vec3f CalculateS(int i, int j);

  parser::Vec3i RenderPixel(int i, int j, const parser::Camera &camera);

public:
  SceneRenderer(const char *scene_path) { scene_.loadFromXml(scene_path); }

  const std::vector<parser::Camera> &Cameras() { return scene_.cameras; }

  parser::Vec3i *RenderImage(const parser::Camera &camera);
};

#endif
