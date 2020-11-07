// Dosya uzun oldugu icin kafam karisti, bu dosyalara boldum simdilik
#include <iostream>
#include "parser.hpp"
#include "ppm.hpp"
#include <math.h>
#include "helper.hpp" // Other includes...
#include "afeser.hpp" // Functions
#include <thread>

#define INFINITE 40000
using namespace std;

// Look afeser.hpp for info..
PrecomputedVariables precomputedVariables;
const int numberOfCores = thread::hardware_concurrency();

typedef struct IntersectionData
{
    
    float t;
    parser::Vec3f normal;
    int materialId;
    
} IntersectionData;

// Global variable'lar neden var cozemedim???
parser::Vec3f *vertexData_PTR;
parser::Scene *scenePTR;

int numberOfSpheres;
int numberOfTriangles;
int numberOfMeshes;


parser::Sphere *spheres;
parser::Triangle *triangles;
parser::Mesh  *meshes;

parser::Camera cam;
parser::Vec3f camUVector;
//int sizeX, sizeY;
float pixelW, pixelH;
float halfPixelW, halfPixelH;




//RGB **image;
unsigned char* immage;

parser::Vec3f cross1(parser::Vec3f a, parser::Vec3f b){
    parser::Vec3f tmp;
    tmp.x =  a.y*b.z-b.y*a.z;
    tmp.y = b.x*a.z-a.x*b.z;
    tmp.z = a.x*b.y-b.x*a.y;
    
    return tmp;
    
}

float lengTh(parser::Vec3f v){
    
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

parser::Vec3f normalize(parser::Vec3f v){
    
    parser::Vec3f tmp;
    float d;
    
    d = lengTh(v);
    tmp.x = v.x/d;
    tmp.y = v.y/d;
    tmp.z = v.z/d;
    
    return tmp;
    
    
}

void freeAll(){
    
    delete[]  spheres;
    delete[]  triangles;
    delete[]  meshes;
    delete[] immage;
    
}


void loadCamera(parser::Camera &x){
    cam.position = x.position;
    cam.gaze = x.gaze;
    cam.up = x.up;
    cam.near_plane = x.near_plane;
    cam.near_distance = x.near_distance;
    cam.image_width = x.image_width;
    cam.image_height = x.image_height;
    cam.image_name = x.image_name;
    
    pixelW = (cam.near_plane.y - cam.near_plane.x)/ (float) cam.image_width;
    halfPixelW = pixelW * 0.5;
    
    pixelH = (cam.near_plane.w - cam.near_plane.z) / (float) cam.image_height;
    halfPixelH = pixelH * 0.5;
    
    camUVector = cross1(cam.gaze, cam.up);
    
    
    camUVector = normalize(camUVector);
    
    cam.up = cross1(camUVector, cam.gaze);
    cam.up = normalize(cam.up);
    
    cam.gaze = normalize(cam.gaze);
    
    int max = cam.image_width * cam.image_height * 3;
    immage =  new unsigned char [ max];
    
    
}


void loadScene(parser::Scene * scene){
    
    
    scenePTR = scene;
    vertexData_PTR = scene->vertex_data.data(); // Bu pointer vertex data vectorunun basina point ediyor.
    
    numberOfSpheres = scene->spheres.size();
    numberOfTriangles = scene->triangles.size();
    numberOfMeshes = scene->meshes.size();
    
    
    spheres = new parser::Sphere[numberOfSpheres];
    triangles = new parser::Triangle[numberOfTriangles];
    meshes = new parser::Mesh[numberOfMeshes];
    
    
    
    for(int i =0; i < numberOfSpheres; i++){
        spheres[i].material_id = scene->spheres[i].material_id;
        spheres[i].center_vertex_id = scene->spheres[i].center_vertex_id;
        spheres[i].radius = scene->spheres[i].radius;
    }
    
    for(int i= 0; i < numberOfTriangles; i++){
        triangles[i].material_id = scene->triangles[i].material_id;
        triangles[i].indices = scene->triangles[i].indices;
    }
    
    for(int i = 0; i <numberOfMeshes; i++){
        meshes[i].material_id = scene->meshes[i].material_id;
        meshes[i].faces = scene->meshes[i].faces;
    }
    
}

parser::Vec3f mult(parser::Vec3f a, float c){
    
    parser::Vec3f tmp;
    
    tmp.x = a.x * c;
    tmp.y = a.y * c;
    tmp.z = a.z * c;
    
    return tmp;
}

parser::Vec3f add(parser::Vec3f a, parser::Vec3f b){
    parser::Vec3f tmp;
    
    tmp.x = a.x + b.x;
    tmp.y = a.y + b.y;
    tmp.z = a.z + b.z;
    
    return tmp;
}


void initImage(parser::Scene *scene){
    
    // Below sets the background color
    int  k =0;
    for(int i =0 ; i < cam.image_width; i++){
        for(int j =0; j < cam.image_height; j++){
            
            
            immage[k++] = (unsigned char) scene->background_color.x;
            immage[k++] = (unsigned char) scene->background_color.y;
            immage[k++] = (unsigned char) scene->background_color.z;
            
            
        }
    }
    
}

parser::Ray generateRay(int i, int j){
    
    parser::Vec3f m;
    parser::Vec3f q;
    parser::Vec3f result;
    float su, sv;
    
    m = add(cam.position, mult(cam.gaze, cam.near_distance));
    q = add(m, add(mult(camUVector,cam.near_plane.x ), mult(cam.up,cam.near_plane.w) ));
    
    su = 0.5* pixelW + j*pixelW;
    sv = 0.5* pixelH + i*pixelH;
    
    result = add(q, add(mult(camUVector, su), mult(cam.up, -sv)));
    
    
    result.x = result.x - cam.position.x;
    result.y = result.y - cam.position.y;
    result.z = result.z - cam.position.z;
    
    result = normalize(result);
    
    parser::Ray ray;
    ray.a = cam.position;
    ray.b = result;
    return ray;
    
}

float intersectSphere(parser::Ray ray, parser::Sphere sphere){ // TODO - neden float bu?? struct yapilari float??
    
    float A, B, C;  // ---> qudritic func constants
    
    float delta;
    
    parser::Vec3f scenter = vertexData_PTR[sphere.center_vertex_id -1];
    
    float sradius = sphere.radius;
    
    
    float t, t1, t2;
    
    C = (ray.a.x-scenter.x)*(ray.a.x-scenter.x) + (ray.a.y-scenter.y)*(ray.a.y-scenter.y) + (ray.a.z-scenter.z)*(ray.a.z-scenter.z) -sradius*sradius;
    
    B = 2*ray.b.x*(ray.a.x-scenter.x) + 2*ray.b.y*(ray.a.y-scenter.y) + 2*ray.b.z*(ray.a.z-scenter.z);
    
    A = ray.b.x*ray.b.x + ray.b.y*ray.b.y + ray.b.z*ray.b.z;
    
    delta = B*B-4*A*C;
    
    if(delta < 0 )
        return -1; // TODO - float comparison bazen yanlis sonuc verebilir sanki??
    else if(delta == 0 ){
        
        t = -B / (2*A);
    }else{
        
        float tmp;
        
        delta = sqrt(delta);
        A = 2*A;
        t1 = (-B + delta) / A;
        t2 = (-B - delta) / A;
        
        if(t2 < t1){
            tmp = t2;
            t2 = t1;
            t1 = tmp;
        }
        
        if(t1 >= 1.0 )
            t = t1;
        else{
            t = t1;
        }
    }
    
    return t;
    
}
parser::Vec3f getAmbientReflectance(int materialID){
    
    return  scenePTR->materials[materialID-1].ambient;
    
}

parser::Vec3f getDiffuseReflectance(int materialID){
    
    return scenePTR->materials[materialID-1].diffuse;
}

parser::Vec3f getspecularReflectance(int materialID){
    
    return scenePTR->materials[materialID-1].specular;
}


float getPhongExponent(int materialID){
    
    return scenePTR->materials[materialID-1].phong_exponent;
}


parser::Vec3f subtract(parser::Vec3f a, parser::Vec3f b){
    
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    
    return a;
}


IntersectionData  intersectRay(parser::Ray ray, float treshold){
    
    
    float tmin = INFINITE;
    
    IntersectionData intersection; // The attributes are the same for all three
    float t1; // To find minimum
    
    for(int k = 0; k < numberOfSpheres; k++){
        
        t1 = intersectSphere(ray, spheres[k]);
        
        if(t1 >= treshold){ // BUNU NEDEN 1 diyoruz ya tam anlamadim?? --> treshold intersection pointin nerde
            //olacagini belirlerken kullaniliyor. 1 den buyuk oldugu durumda image planin gorunmeyen kismiyla ilgilenmiyoruz demek
            
            
            if(t1 < tmin){
                tmin =t1;
                
                
                parser::Vec3f point = add(ray.a, mult(ray.b, tmin)); // find point on the object.
                
                parser::Vec3f sphereCenter = vertexData_PTR[spheres[k].center_vertex_id-1];
                parser::Vec3f normal = normalize(add(point, mult(sphereCenter, -1)));
                
                intersection.materialId = scenePTR->spheres[k].material_id;
                intersection.normal     = normal;
                intersection.t = t1;
                
            }
        }
    }
    
    
    // Intersect triangle
    for(int k = 0; k < scenePTR->triangles.size(); k++){
        
        
        t1 = intersectTriangle(ray, scenePTR->triangles[k].indices, scenePTR->vertex_data);
        
        
        if(t1 >= treshold){
            
            if(t1 < tmin){
                intersection.t          = t1;
                intersection.normal     = (*precomputedVariables.simpleTriangleNormalVectors)[k];
                intersection.materialId = scenePTR->triangles[k].material_id;
                
                tmin = t1;
                
            }
        }
    }
    // Intersect meshes
    for(int k = 0; k < scenePTR->meshes.size(); k++){
        std::pair<int, float> indexAndValueT;
        
        
        // Return the index that is closest and t value of the closes face
        indexAndValueT = intersectMesh(ray, scenePTR->meshes[k].faces, scenePTR->vertex_data, tmin, treshold, k);
        
        
        if(-1 != indexAndValueT.first){
            intersection.t          = indexAndValueT.second;
            intersection.normal     = (*(*precomputedVariables.meshNormalVectors)[k])[indexAndValueT.first];
            intersection.materialId = scenePTR->meshes[k].material_id;
            
            tmin = indexAndValueT.second;
        }
    }
    
    intersection.t = tmin;
    
    return intersection;
    
}


parser::Vec3f computeLightContribution(parser::Vec3f point, parser::PointLight light){
    
    //E(d) == I / d^2
    
    parser::Vec3f lightDirection = subtract(light.position, point);
    float lightDistance = lengTh(lightDirection);
    parser::Vec3f irradianceContribution = vectorDivision(light.intensity, lightDistance * lightDistance);
    return irradianceContribution;
    
}

parser::Vec3f computeDiffuse(int materialID, parser::Vec3f normal, parser::Vec3f normalizedLightDirection, parser::Vec3f irradiance ){
    
    normal = normalize(normal);
    float cosTheta = fmax(0.0f,  dot(normalizedLightDirection, normal));
    
    
    parser::Vec3f diffuse = vectorMultiplication(irradiance, cosTheta);
    parser::Vec3f diffuseReflectance = getDiffuseReflectance(materialID);
    // Multiplying with kd
    diffuse.x *= diffuseReflectance.x;
    diffuse.y *= diffuseReflectance.y;
    diffuse.z *= diffuseReflectance.z;
    
    return diffuse;
}


parser::Vec3f computeSpecular(int materialID, parser::Vec3f normal, parser::Vec3f irradiance, parser::Vec3f normalizedHalfVector ){
    
    float phongEx = scenePTR->materials[materialID-1].phong_exponent;
    float phongExponentCosAlpha =  pow(max(0.0f,(float) dot(normalize(normal), normalizedHalfVector)), phongEx );
    // (cosAlpha)^ns * E(d)
    parser::Vec3f specular = vectorMultiplication(irradiance, phongExponentCosAlpha);
    parser::Vec3f specularReflectance = getspecularReflectance(materialID);
    
    // Multiplying with specular coeff
    specular.x *= specularReflectance.x;
    specular.y *= specularReflectance.y;
    specular.z *= specularReflectance.z;
    
    return specular;
}



parser::Vec3f computeColor( parser::Ray ray, IntersectionData intersection, int recursionNumber,parser::Scene *scene){
    
    parser::Vec3f pixelColor = {};
    int materialID = intersection.materialId;
    parser::Vec3f point = add(ray.a, mult(ray.b, intersection.t)); // find point on the object.
    parser::Vec3f eyeVector = subtract(ray.a, point);
    parser::Vec3f normalizedEyeVector = normalize(eyeVector);
    parser::Vec3f ambientReflectance = getAmbientReflectance(materialID);
    parser::Vec3f ambienLight = scenePTR->ambient_light;
    pixelColor.x = (ambienLight.x * ambientReflectance.x);
    pixelColor.y = (ambienLight.y * ambientReflectance.y);
    pixelColor.z = (ambienLight.z * ambientReflectance.z);
    
    
    for(auto y = scene->point_lights.begin(); y < scene->point_lights.end(); y++){ //Her bir light source icin D ve S hesaplar
        
        parser::Vec3f lightPosition = (*y).position;
        parser::Vec3f lightDirection = add(lightPosition, mult(point, -1));
        parser::Vec3f normalizedLightDirection = normalize(lightDirection); //  L - P = toLight
        parser::Ray shadowRay;
        parser::Vec3f intOffset = mult(normalizedLightDirection, scenePTR->shadow_ray_epsilon);
        shadowRay.a = add(point, intOffset);
        shadowRay.b = normalizedLightDirection;
        float shadowIntersection = intersectRay(shadowRay, 0).t; //treshold 0 cunku belli bir noktanin otesindeki kesisimleri isteme gibi bir sartimiz yok
        
        if(shadowIntersection >= lengTh(lightDirection)){
            
            
            parser::Vec3f irradiance = computeLightContribution(point, (*y));
            parser::Vec3f diffuseContribution = computeDiffuse(materialID,  intersection.normal, normalizedLightDirection, irradiance );
            
            pixelColor = vectorAddition(diffuseContribution, pixelColor); // add diffuse contribution to the pixel color
            
            // compute specular contribution
            parser::Vec3f normalizedHalfVector = normalize(vectorAddition(normalizedLightDirection, normalizedEyeVector));
            parser::Vec3f specularContribution = computeSpecular(materialID, intersection.normal, irradiance, normalizedHalfVector );
            pixelColor = vectorAddition(specularContribution, pixelColor); // add specular contribution to the pixel color
        }
        else{
            
            continue;
        }
        
    }
    
    parser::Vec3f mirrorComponenet = scene->materials[materialID-1].mirror;
    
    if(recursionNumber > 0 && (mirrorComponenet.x > 0 || mirrorComponenet.y > 0 || mirrorComponenet.z > 0)){
        
        parser::Ray reflectedRay;
        float cosTheta = dot(intersection.normal, normalizedEyeVector);
        reflectedRay.b = add(mult(normalizedEyeVector, -1), mult(intersection.normal, 2*cosTheta));
        reflectedRay.a = add(point ,mult(reflectedRay.b, scenePTR->shadow_ray_epsilon) );
        IntersectionData reflectedIntersection = intersectRay(reflectedRay,0); //treshold 0 cunku belli bir noktanin otesindeki kesisimleri isteme gibi bir sartimiz yok
        
        if(reflectedIntersection.t < INFINITE && reflectedIntersection.t > 0){ //  ray hits an object
            
            //parser::Ray ray, IntersectionData intersection, int recursionNumber,parser::Scene *scene
            parser::Vec3f reflectedRadiance = computeColor(reflectedRay, reflectedIntersection, recursionNumber-1, scene);
            reflectedRadiance.x *= mirrorComponenet.x;
            reflectedRadiance.y *= mirrorComponenet.y;
            reflectedRadiance.z *= mirrorComponenet.z;
            pixelColor = add(reflectedRadiance, pixelColor);
        }
    }
    
    pixelColor.x = min(max(0.0f, pixelColor.x), 255.0f);
    pixelColor.y = min(max(0.0f, pixelColor.y), 255.0f);
    pixelColor.z = min(max(0.0f, pixelColor.z), 255.0f);
    return pixelColor;
}


void processImage(parser::Camera cam, parser::Scene *scene, int image_width, int finishHeight, int startingHeight){
    
    for (int i = startingHeight; i < finishHeight; i++){
        std::cout <<"% "<<       ((i -startingHeight)*100)/(finishHeight -startingHeight) <<endl<<std::flush; // Programin hizini intuitive anlamak icin koydum sadece
        for(int j = 0; j < cam.image_width; j++){
            
            parser::Ray ray;
            
            ray = generateRay(i,j);
            
            IntersectionData intersection; // The attributes are the same for all three
            intersection = intersectRay(ray, 1); //treshold 1 cunku tresholdun 1 oldugu nokta image plane 1 den kucuk oldugu noktalardaki kesisimler image planenin arkasinda kalacagindan gorunmeyecekler.
            
            if(intersection.t < INFINITE){
                parser::Vec3f color = computeColor(ray, intersection, scenePTR->max_recursion_depth, scene); // just one function for coloring
                
                
                int wherePixelStarts = i*(cam.image_width)*3 + j*3;
                
                
                immage[wherePixelStarts]    = (unsigned char) color.x;
                immage[wherePixelStarts + 1] = (unsigned char) color.y;
                immage[wherePixelStarts + 2] = (unsigned char) color.z;
            }
        }
    }
    
}




int main(int argc, char* argv[])
{
    // Sample usage for reading an XML scene file
    parser::Scene scene;
    
    scene.loadFromXml(argv[1]);
    
    // Precompute normal vectors
    precomputeAllVariables(scene);
    
    for(auto x = scene.cameras.begin(); x < scene.cameras.end(); x++){
        
        loadCamera(*x);
        loadScene(&scene);
        initImage(&scene);
        
        const int width = (*x).image_width;
        const int height = (*x).image_height;
        if(numberOfCores == 0 || height < numberOfCores){
            processImage( (*x),  &scene, width , height, 0);
        }else{
            thread* threads = new thread[numberOfCores];
            const int heightIncrease = height/numberOfCores;
            
            for(int i = 0; i < numberOfCores; i++){
                const int min_height = i*heightIncrease;
                
                if(i+1 != numberOfCores){
                    const int max_height = (i+1)*heightIncrease;
                    threads[i] = thread(processImage, (*x), &scene, width, max_height, min_height);
                }else{
                    const int max_height = height;
                    threads[i] = thread(processImage, (*x), &scene, width, max_height, min_height);
                }
            }
            
            for(int i =0; i < numberOfCores; i++){
                threads[i].join();
            }
            delete[] threads;
        }
        
        
        // For every camera write another ppm file image.
        write_ppm(cam.image_name.c_str(), immage, cam.image_width, cam.image_height);
    }
    
    // Free memory at the end
    freeNormalVectorMemory();
    freeAll();
    
    return 0;
    
}
