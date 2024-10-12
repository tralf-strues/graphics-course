#pragma once

#include <span>

#include <scene/Camera.hpp>
#include "shaders/Light.h"

/**
 * Contains data sent from the gameplay/logic part of the application
 * to the renderer on every frame.
 */
struct FramePacket
{
  Camera mainCam;
  float currentTime = 0;

  std::span<const PointLight> pointLights;
  DirectionalLight dirLight;
};
