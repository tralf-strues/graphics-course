#pragma once

#include "wsi/OsWindowingManager.hpp"
#include "scene/Camera.hpp"
#include "shaders/Light.h"

#include "Renderer.hpp"


/**
 * Main class of the application. Contains things that are not strictly
 * related to rendering, e.g. OS window creation, input handling.
 */
class App
{
public:
  App();

  void run();

private:
  void processInput(float dt);
  void drawFrame();

  void onGuiFrame();

  void moveCam(Camera& cam, const Keyboard& kb, float dt);
  void rotateCam(Camera& cam, const Mouse& ms, float dt);

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> mainWindow;

  float camMoveSpeed = 1;
  float camRotateSpeed = 0.1f;
  float zoomSensitivity = 2.0f;
  Camera mainCam;
  bool cameraMoved = false;

  DirectionalLight dirLight;
  std::vector<PointLight> pointLights;

  std::unique_ptr<Renderer> renderer;
};
