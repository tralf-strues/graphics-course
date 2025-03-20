include(${PROJECT_SOURCE_DIR}/cmake/common.cmake)

add_executable(demo
  main.cpp
  Renderer.cpp
  EnvironmentManager.cpp
  HiZPass.cpp
  SharpenPass.cpp
  TAAPass.cpp
  WorldRenderer.cpp
  App.cpp
)

target_link_libraries(demo
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(demo
  shaders/bake_diffuse_irradiance.comp
  shaders/compute_env_brdf.comp
  shaders/convert_cubemap.comp
  shaders/cubemap.frag
  shaders/cubemap.vert
  shaders/deferred_pass.comp
  shaders/demo_diffuse_indirect.frag
  shaders/demo_diffuse_sh.frag
  shaders/demo_specular_ibl.frag
  shaders/geometry_pass.vert
  shaders/geometry_pass.frag
  shaders/hiz.comp
  shaders/prefilter_envmap.comp
  shaders/sharpen.comp
  shaders/taa_resolve.comp
)
