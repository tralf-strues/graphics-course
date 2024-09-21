include(${PROJECT_SOURCE_DIR}/cmake/common.cmake)

add_executable(demo
  main.cpp
  Renderer.cpp
  WorldRenderer.cpp
  App.cpp
)

target_link_libraries(demo
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(demo
  shaders/simple.vert
  shaders/simple_shadow.frag
)
