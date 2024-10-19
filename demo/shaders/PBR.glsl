#ifndef PBR_GLSL_INCLUDED
#define PBR_GLSL_INCLUDED

const float PI      = 3.14159265359f;
const float TWO_PI  = 2.0f * PI;
const float HALF_PI = 0.5f * PI;
const float INV_PI  = 1.0f / PI;

struct SurfacePoint
{
  vec3  normal;
  vec3  position;
  vec3  toCam;

  vec3  albedo;
  float metalness;
  float roughness;
  vec3  f0;
};

struct LightSample
{
  vec3  toLight;
  vec3  halfVector;
  vec3  radiance;
};

vec3 LambertianDiffuseBRDF(SurfacePoint point);
vec3 DiffuseBRDF(SurfacePoint point, vec3 fresnel);
vec3 SpecularBRDF(SurfacePoint point, LightSample light, vec3 fresnel);

float NDF_TRGGX(vec3 n, vec3 h, float roughness);
vec3  FresnelSchlick(vec3 h, vec3 v, vec3 f0);
float GSF_SchlickGGX(vec3 n, vec3 v, float k);
float GSF_SmithSchlickGGX(vec3 n, vec3 v, vec3 l, float k);

vec3 CalculateRadiance(SurfacePoint point, LightSample light)
{
  // Fresnel Function
  // aka how much light is reflected
  vec3 fresnel = FresnelSchlick(light.halfVector, point.toCam, point.f0);

  vec3 BRDF     = DiffuseBRDF(point, fresnel) + SpecularBRDF(point, light, fresnel);
  vec3 radiance = BRDF * light.radiance * max(dot(point.normal, light.toLight), 0.0f);

  return radiance;
}

vec3 LambertianDiffuseBRDF(SurfacePoint point)
{
    return point.albedo * INV_PI;
}

vec3 DiffuseBRDF(SurfacePoint point, vec3 fresnel)
{
  vec3 Kd = (1.0f - fresnel) * (1.0f - point.metalness);
  return Kd * LambertianDiffuseBRDF(point);
}

vec3 SpecularBRDF(SurfacePoint point, LightSample light, vec3 fresnel)
{
  // Normal Distribution Function
  // aka alignment of microfacets to the halfway vector
  float NDF                 = NDF_TRGGX(point.normal, light.halfVector, point.roughness);

  // Geometry Shadowing Function
  // aka coefficient on the amount of light, which is self-shadowed or obstructed
  float k                   = (point.roughness + 1.0f) *
                               (point.roughness + 1.0f) / 8.0f;  // Roughness remapping
  float GSF                 = GSF_SmithSchlickGGX(point.normal, point.toCam, light.toLight, k);

  // Specular part
  vec3  Ks                  = fresnel;
  float specularNumerator   = NDF * GSF;
  float specularDenominator = 4.0f *
                              max(dot(point.normal, point.toCam),   0.0f) *
                              max(dot(point.normal, light.toLight), 0.0f) +
                              0.0001f; // Used to avoid division by zero
  float specular            = specularNumerator / specularDenominator;

  return Ks * specular;
}

float NDF_TRGGX(vec3 n, vec3 h, float roughness)
{
  float ggxAlpha    = roughness * roughness;  // Apparently looks better
  float ggxAlpha2   = ggxAlpha * ggxAlpha;

  float nh          = max(dot(n, h), 0.0f);

  float numerator   = ggxAlpha2;
  float denominator = nh * nh * (ggxAlpha2 - 1.0) + 1.0;
        denominator = PI * denominator * denominator;

  return numerator / denominator;
}

vec3 FresnelSchlick(vec3 h, vec3 v, vec3 f0)
{
  return f0 + (1.0 - f0) * pow(clamp(1.0f - max(dot(h, v), 0.0f), 0.0f, 1.0f), 5.0f);
}

float GSF_SchlickGGX(vec3 n, vec3 v, float k)
{
  float nv = max(dot(n, v), 0.0f);
  return nv / (nv * (1.0f - k) + k);
}

float GSF_SmithSchlickGGX(vec3 n, vec3 v, vec3 l, float k)
{
  float shadowing   = GSF_SchlickGGX(n, l, k);
  float obstruction = GSF_SchlickGGX(n, v, k);

  return shadowing * obstruction;
}

#endif // PBR_GLSL_INCLUDED
