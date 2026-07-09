#include "pipeline.hpp"

#include "../webgpu/gpu.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "tracy/Tracy.hpp"

#include <cstdlib>

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

namespace {
enum class Group0Binding : u8 {
  Static,
  VertexTexture,
};

Group0Binding s_boundGroup0 = Group0Binding::Static;
bool s_boundTextureGroupEmpty = true;
gfx::BindGroupRef s_boundTextureGroup = 0;
bool s_boundFullIndexBuffer = false;

bool portmaster_bind_full_index_buffer_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("DUSKLIGHT_PORTMASTER_BIND_FULL_INDEX_BUFFER");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}
} // namespace

void reset_render_state() noexcept {
  s_boundGroup0 = Group0Binding::Static;
  s_boundTextureGroupEmpty = true;
  s_boundTextureGroup = 0;
  s_boundFullIndexBuffer = false;
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }

  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, gfx::g_uniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    if (s_boundTextureGroupEmpty || s_boundTextureGroup != data.bindGroups.textureBindGroup) {
      pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
      s_boundTextureGroupEmpty = false;
      s_boundTextureGroup = data.bindGroups.textureBindGroup;
    }
  }
  const auto desiredGroup0 = data.textureVertexFetch ? Group0Binding::VertexTexture : Group0Binding::Static;
  if (s_boundGroup0 != desiredGroup0) {
    pass.SetBindGroup(0, data.textureVertexFetch ? gfx::g_vertexTextureBindGroup : gfx::g_staticBindGroup);
    s_boundGroup0 = desiredGroup0;
  }
  if (data.nativeVertexFetch) {
    pass.SetVertexBuffer(0, gfx::g_vertexBuffer, data.vertRange.offset, data.vertRange.size);
  }
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  if (data.idxRange.size == 0) {
    pass.Draw(data.indexCount, data.instanceCount);
  } else if (data.textureVertexFetch && portmaster_bind_full_index_buffer_enabled()) {
    if (!s_boundFullIndexBuffer) {
      pass.SetIndexBuffer(gfx::g_indexBuffer, wgpu::IndexFormat::Uint16, 0, gfx::IndexBufferSize);
      s_boundFullIndexBuffer = true;
    }
    pass.DrawIndexed(data.indexCount, data.instanceCount, data.idxRange.offset / sizeof(u16));
  } else {
    pass.SetIndexBuffer(gfx::g_indexBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
    s_boundFullIndexBuffer = false;
    pass.DrawIndexed(data.indexCount, data.instanceCount);
  }
}
} // namespace aurora::gx
