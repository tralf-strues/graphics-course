// Reference: https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
uint WangHash(inout uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);

  return seed;
}

float RandomFloatNormalized(inout uint seed) {
  return float(WangHash(seed)) / 4294967296.0f;
}

vec3 RandomUnitVector(inout uint seed) {
  float z = RandomFloatNormalized(seed) * 2.0f - 1.0f;
  float a = RandomFloatNormalized(seed) * TWO_PI;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);

  return vec3(x, y, z);
}
