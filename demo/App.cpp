#include "App.hpp"

#include <tracy/Tracy.hpp>
#include <imgui.h>

#include "gui/ImGuiRenderer.hpp"


App::App()
{
  glm::uvec2 initialRes = {1280, 720};
  mainWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = initialRes,
    .resizeable = true,
    .refreshCb =
      [this]() {
        // NOTE: this is only called when the window is being resized.
        drawFrame();
        FrameMark;
      },
    .resizeCb =
      [this](glm::uvec2 res) {
        if (res.x == 0 || res.y == 0)
          return;

        renderer->recreateSwapchain(res);
      },
  });

  renderer.reset(new Renderer(initialRes));

  auto instExts = windowing.getRequiredVulkanInstanceExtensions();
  renderer->initVulkan(instExts);

  auto surface = mainWindow->createVkSurface(etna::get_context().getInstance());

  renderer->initFrameDelivery(
    std::move(surface), [window = mainWindow.get()]() { return window->getResolution(); });

  // TODO: this is bad design, this initialization is dependent on the current ImGui context, but we
  // pass it implicitly here instead of explicitly. Beware if trying to do something tricky.
  ImGuiRenderer::enableImGuiForWindow(mainWindow->native());

  mainCam.lookAt({0.0f, 3.5f, 5.0f}, {0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f});

  dirLight.radiance = glm::vec3(1.0f) * 3.0f;
  dirLight.direction = normalize(glm::vec3(-1.0f, -10.0f, -8.0f));

  pointLights.resize(5);

  // renderer->loadScene(GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/white_sphere/white_sphere.gltf");
  // renderer->loadScene(GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/DamagedHelmet/DamagedHelmet.gltf");
  renderer->loadScene(GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/DemoScene/demo_scene.gltf");
}

void App::run()
{
  double lastTime = windowing.getTime();
  while (!mainWindow->isBeingClosed())
  {
    const double currTime = windowing.getTime();
    const float diffTime = static_cast<float>(currTime - lastTime);
    lastTime = currTime;

    windowing.poll();

    processInput(diffTime);

    const auto pointLightCount = static_cast<uint32_t>(pointLights.size());
    const auto invPointLightCount = 1.0f / static_cast<float>(pointLightCount);

    for (uint32_t i = 0; i < pointLightCount; ++i)
    {
      float alpha = 0.5f * static_cast<float>(currTime) +
        2.0f * static_cast<float>(i) * glm::pi<float>() * invPointLightCount;

      auto& light = pointLights[i];
      light.radiance = glm::vec3(1.0f, 0.6f, 0.3f) * 2.0f;
      light.radius = 7.0f;
      light.position.x = 3.0f * std::cos(alpha);
      light.position.y = 0.0f;
      light.position.z = 3.0f * std::sin(alpha);
    }

    drawFrame();

    FrameMark;
  }
}

void App::processInput(float dt)
{
  ZoneScoped;

  cameraMoved = false;

  if (mainWindow->keyboard[KeyboardKey::kEscape] == ButtonState::Falling)
    mainWindow->askToClose();

  if (is_held_down(mainWindow->keyboard[KeyboardKey::kLeftShift]))
    camMoveSpeed = 10;
  else
    camMoveSpeed = 1;

  if (mainWindow->mouse[MouseButton::mbRight] == ButtonState::Rising)
    mainWindow->captureMouse = !mainWindow->captureMouse;

  moveCam(mainCam, mainWindow->keyboard, dt);
  if (mainWindow->captureMouse)
    rotateCam(mainCam, mainWindow->mouse, dt);

  renderer->debugInput(mainWindow->keyboard);
}

void App::drawFrame()
{
  ZoneScoped;

  {
    ZoneScopedN("drawGui");
    renderer->beginGuiFrame();

    ImGui::Begin("Settings");
    onGuiFrame();
    renderer->onGuiFrame();
    ImGui::End();

    renderer->endGuiFrame();
  }

  renderer->update(FramePacket{
    .mainCam = mainCam,
    .currentTime = static_cast<float>(windowing.getTime()),
    .cameraMoved = cameraMoved,
    .pointLights = pointLights,
    .dirLight = dirLight,
  });
  renderer->drawFrame();
}

void App::onGuiFrame()
{
  if (ImGui::CollapsingHeader("Application", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::Text(
      "Application average %.3f ms/frame (%.1f FPS)",
      1000.0f / ImGui::GetIO().Framerate,
      ImGui::GetIO().Framerate);

    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Press 'B' to recompile and reload shaders");
    ImGui::NewLine();
  }
}

void App::moveCam(Camera& cam, const Keyboard& kb, float dt)
{
  // Move position of camera based on WASD keys, and FR keys for up and down

  glm::vec3 dir = {0, 0, 0};

  if (is_held_down(kb[KeyboardKey::kS]))
  {
    dir -= cam.forward();
    cameraMoved = true;
  }

  if (is_held_down(kb[KeyboardKey::kW]))
  {
    dir += cam.forward();
    cameraMoved = true;
  }

  if (is_held_down(kb[KeyboardKey::kA]))
  {
    dir -= cam.right();
    cameraMoved = true;
  }

  if (is_held_down(kb[KeyboardKey::kD]))
  {
    dir += cam.right();
    cameraMoved = true;
  }

  // if (is_held_down(kb[KeyboardKey::kF]))
  // {
  //   dir -= cam.up();
  //   cameraMoved = true;
  // }

  // if (is_held_down(kb[KeyboardKey::kR]))
  // {
  //   dir += cam.up();
  //   cameraMoved = true;
  // }

  // NOTE: This is how you make moving diagonally not be faster than
  // in a straight line.
  cam.move(dt * camMoveSpeed * (length(dir) > 1e-9 ? normalize(dir) : dir));
}

void App::rotateCam(Camera& cam, const Mouse& ms, float /*dt*/)
{
  if (
    std::abs(ms.capturedPosDelta.x) > 0.0001f || std::abs(ms.capturedPosDelta.y) > 0.0001f ||
    std::abs(ms.scrollDelta.y) > 0.0001f)
  {
    cameraMoved = true;
  }

  // Rotate camera based on mouse movement
  cam.rotate(camRotateSpeed * ms.capturedPosDelta.y, camRotateSpeed * ms.capturedPosDelta.x);

  // Increase or decrease field of view based on mouse wheel
  cam.fov -= zoomSensitivity * ms.scrollDelta.y;
  if (cam.fov < 1.0f)
    cam.fov = 1.0f;
  if (cam.fov > 120.0f)
    cam.fov = 120.0f;
}
