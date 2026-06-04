#include "pipeline.hpp"

#include "../webgpu/gpu.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora::gx {
static Module Log("aurora::gx");

struct NativeVertexLayout {
  std::array<wgpu::VertexAttribute, MaxVtxAttr> attrs;
  std::array<wgpu::VertexBufferLayout, 1> buffers;
};

static NativeVertexLayout native_vertex_layout(const PipelineConfig& config) {
  NativeVertexLayout out{};
  uint32_t attributeCount = 0;
  const auto add = [&](GXAttr attr, wgpu::VertexFormat format) {
    if (config.shaderConfig.attrs[attr].attrType == GX_NONE) {
      return;
    }
    const uint32_t location = attributeCount++;
    out.attrs[location] = wgpu::VertexAttribute{
        .format = format,
        .offset = config.shaderConfig.attrs[attr].offset,
        .shaderLocation = location,
    };
  };
  add(GX_VA_PNMTXIDX, wgpu::VertexFormat::Uint32);
  for (GXAttr attr = GX_VA_TEX0MTXIDX; attr <= GX_VA_TEX7MTXIDX; attr = static_cast<GXAttr>(attr + 1)) {
    add(attr, wgpu::VertexFormat::Uint32);
  }
  add(GX_VA_POS, wgpu::VertexFormat::Float32x3);
  add(GX_VA_NRM, wgpu::VertexFormat::Float32x3);
  add(GX_VA_CLR0, wgpu::VertexFormat::Unorm8x4);
  add(GX_VA_CLR1, wgpu::VertexFormat::Unorm8x4);
  for (GXAttr attr = GX_VA_TEX0; attr <= GX_VA_TEX7; attr = static_cast<GXAttr>(attr + 1)) {
    add(attr, wgpu::VertexFormat::Float32x2);
  }
  out.buffers = {
      wgpu::VertexBufferLayout{
          .stepMode = wgpu::VertexStepMode::Vertex,
          .arrayStride = config.shaderConfig.vtxStride,
          .attributeCount = attributeCount,
          .attributes = out.attrs.data(),
      },
  };
  return out;
}

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto shader = build_shader(config.shaderConfig);
  if (config.shaderConfig.nativeVertexFetch) {
    const auto layout = native_vertex_layout(config);
    return build_pipeline(config, layout.buffers, shader, "GX Pipeline");
  }
  return build_pipeline(config, {}, shader, "GX Pipeline");
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }

  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, gfx::g_uniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
  }
  if (data.nativeVertexFetch) {
    pass.SetVertexBuffer(0, gfx::g_vertexBuffer, data.vertRange.offset, data.vertRange.size);
  }
  pass.SetIndexBuffer(gfx::g_indexBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  pass.DrawIndexed(data.indexCount, data.instanceCount);
}
} // namespace aurora::gx
