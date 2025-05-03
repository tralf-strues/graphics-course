#include "Utils.hpp"

void generate_mips(vk::CommandBuffer& cmds, etna::Image& img, uint32_t layers)
{
  auto extent = img.getExtent();
  auto mips =
    static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;

  if (mips == 1)
  {
    return;
  }

  /* Transition mip 0 to transfer src */
  etna::set_state(
    cmds,
    img.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferRead,
    vk::ImageLayout::eTransferSrcOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmds);

  /* Transition mips [1, mips - 1] to transfer dst */
  vk::ImageMemoryBarrier barrier;
  barrier.setImage(img.get());
  barrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
  barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  barrier.setOldLayout(vk::ImageLayout::eUndefined);
  barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
  barrier.setSubresourceRange(vk::ImageSubresourceRange{
    vk::ImageAspectFlagBits::eColor,
    /*baseMip=*/1,
    /*levelCount=*/mips - 1,
    0,
    layers,
  });

  cmds.pipelineBarrier(
    vk::PipelineStageFlagBits::eTopOfPipe,
    vk::PipelineStageFlagBits::eTransfer,
    {},
    {},
    {},
    barrier);

  for (uint32_t mip = 1; mip < mips; ++mip) {
    uint32_t mipWidth = extent.width >> mip;
    uint32_t mipHeight = extent.height >> mip;

    vk::ImageBlit blit;
    blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.srcOffsets[1] = vk::Offset3D{
      static_cast<int32_t>(mipWidth << 1),
      static_cast<int32_t>(mipHeight << 1),
      1,
    };

    blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.dstOffsets[1] = vk::Offset3D{
      static_cast<int32_t>(mipWidth),
      static_cast<int32_t>(mipHeight),
      1,
    };

    blit.setSrcSubresource({vk::ImageAspectFlagBits::eColor, mip - 1, 0, layers});
    blit.setDstSubresource({vk::ImageAspectFlagBits::eColor, mip, 0, layers});

    cmds.blitImage(
      img.get(),
      vk::ImageLayout::eTransferSrcOptimal,
      img.get(),
      vk::ImageLayout::eTransferDstOptimal,
      blit,
      vk::Filter::eLinear);

    /* Transition mip to transfer src */
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    barrier.setSubresourceRange(vk::ImageSubresourceRange{
      vk::ImageAspectFlagBits::eColor,
      /*baseMip=*/mip,
      /*levelCount=*/1,
      0,
      layers,
    });

    cmds.pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eTransfer,
      {},
      {},
      {},
      barrier);
  }
}

void generate_mips(vk::CommandBuffer& cmds, etna::Image& src, etna::Image& dst)
{
  auto extent = src.getExtent();
  auto mips =
    static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;

  /* Transition src image's mip 0 to transfer src */
  etna::set_state(
    cmds,
    src.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferRead,
    vk::ImageLayout::eTransferSrcOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmds,
    dst.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite,
    vk::ImageLayout::eTransferDstOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmds);

  /* Transition dst image's mips [0, mips - 1] to transfer dst */
  vk::ImageMemoryBarrier barrier;
  barrier.setImage(dst.get());
  barrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
  barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  barrier.setOldLayout(vk::ImageLayout::eUndefined);
  barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
  barrier.setSubresourceRange(vk::ImageSubresourceRange{
    vk::ImageAspectFlagBits::eColor,
    /*baseMip=*/0,
    /*levelCount=*/mips,
    0,
    1,
  });

  int32_t srcWidth = static_cast<int32_t>(extent.width);
  int32_t srcHeight = static_cast<int32_t>(extent.height);
  int32_t dstWidth = srcWidth;
  int32_t dstHeight = srcHeight;

  for (uint32_t dstMip = 0; dstMip < mips; ++dstMip)
  {
    uint32_t srcMip = dstMip == 0 ? 0 : dstMip - 1;

    vk::ImageBlit blit;
    blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.srcOffsets[1] = vk::Offset3D{srcWidth, srcHeight, 1};

    blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.dstOffsets[1] = vk::Offset3D{dstWidth, dstHeight, 1};

    blit.setSrcSubresource({vk::ImageAspectFlagBits::eColor, srcMip, 0, 1});
    blit.setDstSubresource({vk::ImageAspectFlagBits::eColor, dstMip, 0, 1});

    cmds.blitImage(
      (dstMip == 0) ? src.get() : dst.get(),
      vk::ImageLayout::eTransferSrcOptimal,
      dst.get(),
      vk::ImageLayout::eTransferDstOptimal,
      blit,
      vk::Filter::eLinear);

    /* Transition dst image's mip to transfer src */
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    barrier.setSubresourceRange(vk::ImageSubresourceRange{
      vk::ImageAspectFlagBits::eColor,
      /*baseMip=*/dstMip,
      /*levelCount=*/1,
      0,
      1,
    });

    cmds.pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eTransfer,
      {},
      {},
      {},
      barrier);

    if (dstMip != 0)
    {
      srcWidth = std::max(srcWidth >> 1, 1);
      srcHeight = std::max(srcHeight >> 1, 1);
    }

    dstWidth = std::max(dstWidth >> 1, 1);
    dstHeight = std::max(dstHeight >> 1, 1);
  }

  etna::set_state_external(
    dst.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferRead,
    vk::ImageLayout::eTransferSrcOptimal);
}