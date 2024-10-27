#pragma once

#include <etna/Vulkan.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>


void generate_mips(vk::CommandBuffer& cmds, etna::Image& img);
