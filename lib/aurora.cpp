#include <aurora/aurora.h>

#ifdef AURORA_ENABLE_GX
#include "gfx/common.hpp"
#include "gx/fifo.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
#include "internal.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
Module Log("aurora");

using Clock = std::chrono::steady_clock;

static bool portmaster_timing_enabled() noexcept {
  const char* disabled = std::getenv("DUSKLIGHT_PORTMASTER_TIMING");
  if (disabled != nullptr && disabled[0] == '0') {
    return false;
  }
  return std::getenv("DUSKLIGHT_PORTMASTER_FBDEV_PRESENT") != nullptr ||
         std::getenv("DUSKLIGHT_PORTMASTER_EGL_FBDEV_SURFACE") != nullptr;
}

static uint64_t elapsed_us(Clock::time_point start, Clock::time_point end = Clock::now()) noexcept {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

#ifdef AURORA_ENABLE_GX
struct PortmasterEndFrameTiming {
  Clock::time_point intervalStart = Clock::now();
  uint64_t frames = 0;
  uint64_t drainUs = 0;
  uint64_t gfxEndUs = 0;
  uint64_t renderEncodeUs = 0;
  uint64_t finishUs = 0;
  uint64_t submitUs = 0;
  uint64_t afterSubmitUs = 0;
  uint64_t finishSubmitUs = 0;
  uint64_t fbdevPresentUs = 0;
  uint64_t totalUs = 0;
  uint64_t maxTotalUs = 0;
};

static PortmasterEndFrameTiming s_portmasterEndFrameTiming{};

static void note_portmaster_end_frame_timing(uint64_t drainUs, uint64_t gfxEndUs, uint64_t renderEncodeUs,
                                             uint64_t finishUs, uint64_t submitUs, uint64_t afterSubmitUs,
                                             uint64_t finishSubmitUs, uint64_t fbdevPresentUs,
                                             uint64_t totalUs) noexcept {
  if (!portmaster_timing_enabled()) {
    return;
  }
  auto& t = s_portmasterEndFrameTiming;
  ++t.frames;
  t.drainUs += drainUs;
  t.gfxEndUs += gfxEndUs;
  t.renderEncodeUs += renderEncodeUs;
  t.finishUs += finishUs;
  t.submitUs += submitUs;
  t.afterSubmitUs += afterSubmitUs;
  t.finishSubmitUs += finishSubmitUs;
  t.fbdevPresentUs += fbdevPresentUs;
  t.totalUs += totalUs;
  t.maxTotalUs = std::max(t.maxTotalUs, totalUs);
  const auto now = Clock::now();
  const auto intervalMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - t.intervalStart).count();
  if (intervalMs < 5000) {
    return;
  }
  const double frames = static_cast<double>(std::max<uint64_t>(1, t.frames));
  Log.info("PortMaster end_frame timing: frames={} avg_total_ms={:.1f} max_total_ms={:.1f} avg_drain_ms={:.1f} avg_gfx_end_ms={:.1f} avg_render_encode_ms={:.1f} avg_finish_ms={:.1f} avg_submit_ms={:.1f} avg_after_submit_ms={:.1f} avg_finish_submit_ms={:.1f} avg_fbdev_present_ms={:.1f}",
           t.frames, static_cast<double>(t.totalUs) / frames / 1000.0, static_cast<double>(t.maxTotalUs) / 1000.0,
           static_cast<double>(t.drainUs) / frames / 1000.0, static_cast<double>(t.gfxEndUs) / frames / 1000.0,
           static_cast<double>(t.renderEncodeUs) / frames / 1000.0, static_cast<double>(t.finishUs) / frames / 1000.0,
           static_cast<double>(t.submitUs) / frames / 1000.0, static_cast<double>(t.afterSubmitUs) / frames / 1000.0,
           static_cast<double>(t.finishSubmitUs) / frames / 1000.0,
           static_cast<double>(t.fbdevPresentUs) / frames / 1000.0);
  t = {};
  t.intervalStart = now;
}
#endif

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;

namespace portmaster_fbdev {
struct State {
  bool initialized = false;
  bool available = false;
  bool singlePage = false;
  int fd = -1;
  uint8_t* pixels = nullptr;
  size_t bytes = 0;
  fb_var_screeninfo var{};
  fb_fix_screeninfo fix{};
  wgpu::Buffer readback;
  uint32_t readbackWidth = 0;
  uint32_t readbackHeight = 0;
  uint32_t readbackStride = 0;
};

State g_state;
std::atomic_bool g_mapDone = false;
bool g_mapOk = false;
static Clock::time_point s_fbdevIntervalStart = Clock::now();
static uint64_t s_fbdevFrames = 0;
static uint64_t s_fbdevWaitUs = 0;
static uint64_t s_fbdevMapUs = 0;
static uint64_t s_fbdevWriteUs = 0;
static uint64_t s_fbdevMaxWaitUs = 0;

void note_fbdev_timing(uint64_t mapUs, uint64_t waitUs, uint64_t writeUs) noexcept {
  if (!portmaster_timing_enabled()) {
    return;
  }
  ++s_fbdevFrames;
  s_fbdevMapUs += mapUs;
  s_fbdevWaitUs += waitUs;
  s_fbdevWriteUs += writeUs;
  s_fbdevMaxWaitUs = std::max(s_fbdevMaxWaitUs, waitUs);
  const auto now = Clock::now();
  const auto intervalMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - s_fbdevIntervalStart).count();
  if (intervalMs < 5000) {
    return;
  }
  const double frames = static_cast<double>(std::max<uint64_t>(1, s_fbdevFrames));
  Log.info("PortMaster fbdev timing: frames={} avg_map_ms={:.1f} avg_wait_ms={:.1f} max_wait_ms={:.1f} avg_write_ms={:.1f}",
           s_fbdevFrames, static_cast<double>(s_fbdevMapUs) / frames / 1000.0,
           static_cast<double>(s_fbdevWaitUs) / frames / 1000.0, static_cast<double>(s_fbdevMaxWaitUs) / 1000.0,
           static_cast<double>(s_fbdevWriteUs) / frames / 1000.0);
  s_fbdevIntervalStart = now;
  s_fbdevFrames = 0;
  s_fbdevMapUs = 0;
  s_fbdevWaitUs = 0;
  s_fbdevWriteUs = 0;
  s_fbdevMaxWaitUs = 0;
}

bool enabled() noexcept {
  return std::getenv("DUSKLIGHT_PORTMASTER_FBDEV_PRESENT") != nullptr;
}

uint32_t present_interval() noexcept {
  const char* value = std::getenv("DUSKLIGHT_PORTMASTER_PRESENT_INTERVAL");
  if (value == nullptr || value[0] == '\0') {
    return 2;
  }
  const auto interval = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
  return std::max(1u, interval);
}

bool should_present_frame() noexcept {
  if (!enabled()) {
    return false;
  }
  static uint32_t frame = 0;
  const uint32_t interval = present_interval();
  const bool present = (frame % interval) == 0;
  ++frame;
  return present;
}

uint32_t align_to(uint32_t value, uint32_t alignment) noexcept {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

bool initialize() noexcept {
  if (g_state.initialized) {
    return g_state.available;
  }
  g_state.initialized = true;
  g_state.fd = open("/dev/fb0", O_RDWR);
  if (g_state.fd < 0) {
    Log.warn("PortMaster fbdev presenter failed to open /dev/fb0: {}", std::strerror(errno));
    return false;
  }
  if (ioctl(g_state.fd, FBIOGET_VSCREENINFO, &g_state.var) != 0 ||
      ioctl(g_state.fd, FBIOGET_FSCREENINFO, &g_state.fix) != 0) {
    Log.warn("PortMaster fbdev presenter failed to query /dev/fb0: {}", std::strerror(errno));
    close(g_state.fd);
    g_state.fd = -1;
    return false;
  }
  if (g_state.var.yres_virtual >= g_state.var.yres) {
    auto pan = g_state.var;
    pan.xoffset = 0;
    pan.yoffset = 0;
    if (ioctl(g_state.fd, FBIOPAN_DISPLAY, &pan) == 0 && ioctl(g_state.fd, FBIOGET_VSCREENINFO, &g_state.var) == 0 &&
        g_state.var.yoffset == 0) {
      g_state.singlePage = true;
    }
  }
  const uint32_t virtualHeight = std::max(g_state.var.yres, g_state.var.yres_virtual);
  g_state.bytes = static_cast<size_t>(g_state.fix.line_length) * virtualHeight;
  g_state.pixels = static_cast<uint8_t*>(mmap(nullptr, g_state.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g_state.fd, 0));
  if (g_state.pixels == MAP_FAILED) {
    Log.warn("PortMaster fbdev presenter failed to mmap /dev/fb0: {}", std::strerror(errno));
    g_state.pixels = nullptr;
    close(g_state.fd);
    g_state.fd = -1;
    return false;
  }
  g_state.available = true;
  Log.warn("PortMaster fbdev presenter active: {}x{} virtual {}x{} offset {}x{} {}bpp stride {} pages {}",
           g_state.var.xres, g_state.var.yres, g_state.var.xres_virtual, g_state.var.yres_virtual, g_state.var.xoffset,
           g_state.var.yoffset, g_state.var.bits_per_pixel, g_state.fix.line_length, g_state.singlePage ? 1 : 2);
  return true;
}

void shutdown() noexcept {
  g_state.readback = {};
  if (g_state.pixels != nullptr) {
    munmap(g_state.pixels, g_state.bytes);
    g_state.pixels = nullptr;
  }
  if (g_state.fd >= 0) {
    close(g_state.fd);
    g_state.fd = -1;
  }
  g_state = {};
}

bool ensure_readback(uint32_t width, uint32_t height) {
  const uint32_t stride = align_to(width * 4u, 256u);
  if (g_state.readback && g_state.readbackWidth == width && g_state.readbackHeight == height &&
      g_state.readbackStride == stride) {
    return true;
  }
  const wgpu::BufferDescriptor descriptor{
      .label = "PortMaster fbdev readback",
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
      .size = static_cast<uint64_t>(stride) * height,
  };
  g_state.readback = g_device.CreateBuffer(&descriptor);
  g_state.readbackWidth = width;
  g_state.readbackHeight = height;
  g_state.readbackStride = stride;
  return g_state.readback != nullptr;
}

void enqueue_readback(const wgpu::CommandEncoder& encoder, const webgpu::TextureWithSampler& source) {
  if (!enabled() || !initialize()) {
    return;
  }
  if (!source.texture || source.size.width == 0 || source.size.height == 0 || !ensure_readback(source.size.width, source.size.height)) {
    return;
  }
  const wgpu::TexelCopyTextureInfo src{
      .texture = source.texture,
  };
  const wgpu::TexelCopyBufferInfo dst{
      .layout =
          wgpu::TexelCopyBufferLayout{
              .offset = 0,
              .bytesPerRow = g_state.readbackStride,
              .rowsPerImage = source.size.height,
          },
      .buffer = g_state.readback,
  };
  const wgpu::Extent3D size{
      .width = source.size.width,
      .height = source.size.height,
      .depthOrArrayLayers = 1,
  };
  encoder.CopyTextureToBuffer(&src, &dst, &size);
}

void write_fb(const uint8_t* src) noexcept {
  const uint32_t srcWidth = g_state.readbackWidth;
  const uint32_t srcHeight = g_state.readbackHeight;
  const uint32_t dstWidth = g_state.var.xres;
  const uint32_t dstHeight = g_state.var.yres;
  if (src == nullptr || dstWidth == 0 || dstHeight == 0 || srcWidth == 0 || srcHeight == 0) {
    return;
  }
  const uint32_t bytesPerPixel = g_state.var.bits_per_pixel / 8u;
  if (bytesPerPixel != 4u && bytesPerPixel != 2u) {
    return;
  }
  const uint32_t virtualHeight = std::max(g_state.var.yres, g_state.var.yres_virtual);
  const uint32_t pageCount = g_state.singlePage ? 1u : std::max(1u, virtualHeight / dstHeight);
  if (bytesPerPixel == 4u && srcWidth == dstWidth && srcHeight == dstHeight) {
    for (uint32_t page = 0; page < pageCount; ++page) {
      uint8_t* dstPage = g_state.pixels + static_cast<size_t>(page * dstHeight) * g_state.fix.line_length;
      for (uint32_t y = 0; y < dstHeight; ++y) {
        const uint8_t* srcRow = src + static_cast<size_t>(y) * g_state.readbackStride;
        auto* dstRow = reinterpret_cast<uint32_t*>(dstPage + static_cast<size_t>(y) * g_state.fix.line_length);
        for (uint32_t x = 0; x < dstWidth; ++x) {
          const uint8_t* p = srcRow + static_cast<size_t>(x) * 4u;
          dstRow[x] = 0xff000000u | (static_cast<uint32_t>(p[0]) << 16u) | (static_cast<uint32_t>(p[1]) << 8u) |
                      static_cast<uint32_t>(p[2]);
        }
      }
    }
    return;
  }
  for (uint32_t y = 0; y < dstHeight; ++y) {
    const uint32_t sy = std::min(srcHeight - 1u, static_cast<uint32_t>((static_cast<uint64_t>(y) * srcHeight) / dstHeight));
    const uint8_t* srcRow = src + static_cast<size_t>(sy) * g_state.readbackStride;
    for (uint32_t page = 0; page < pageCount; ++page) {
      uint8_t* dstRow = g_state.pixels + static_cast<size_t>(page * dstHeight + y) * g_state.fix.line_length;
      for (uint32_t x = 0; x < dstWidth; ++x) {
        const uint32_t sx =
            std::min(srcWidth - 1u, static_cast<uint32_t>((static_cast<uint64_t>(x) * srcWidth) / dstWidth));
        const uint8_t* p = srcRow + static_cast<size_t>(sx) * 4u;
        const uint8_t r = p[0];
        const uint8_t g = p[1];
        const uint8_t b = p[2];
        if (bytesPerPixel == 4u) {
          uint8_t* d = dstRow + static_cast<size_t>(x) * 4u;
          d[0] = b;
          d[1] = g;
          d[2] = r;
          d[3] = 0xff;
        } else {
          auto* d = reinterpret_cast<uint16_t*>(dstRow + static_cast<size_t>(x) * 2u);
          *d = static_cast<uint16_t>(((r >> 3u) << 11u) | ((g >> 2u) << 5u) | (b >> 3u));
        }
      }
    }
  }
}

void present_after_submit() {
  if (!enabled() || !g_state.available || !g_state.readback) {
    return;
  }
  g_mapDone.store(false, std::memory_order_release);
  g_mapOk = false;
  const uint64_t byteSize = static_cast<uint64_t>(g_state.readbackStride) * g_state.readbackHeight;
  const auto mapStart = Clock::now();
  const auto future = g_state.readback.MapAsync(
      wgpu::MapMode::Read, 0, byteSize, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (status != wgpu::MapAsyncStatus::Success) {
          Log.warn("PortMaster fbdev readback map failed: {} {}", magic_enum::enum_name(status), message);
        }
        g_mapOk = status == wgpu::MapAsyncStatus::Success;
        g_mapDone.store(true, std::memory_order_release);
      });
  const auto mapQueued = Clock::now();
  const auto status = webgpu::g_instance.WaitAny(future, 1000000000);
  const auto waitDone = Clock::now();
  if (status != wgpu::WaitStatus::Success || !g_mapDone.load(std::memory_order_acquire) || !g_mapOk) {
    Log.warn("PortMaster fbdev readback wait failed: {}", magic_enum::enum_name(status));
    return;
  }
  const auto* data = static_cast<const uint8_t*>(g_state.readback.GetConstMappedRange(0, byteSize));
  const auto writeStart = Clock::now();
  write_fb(data);
  const auto writeDone = Clock::now();
  note_fbdev_timing(elapsed_us(mapStart, mapQueued), elapsed_us(mapQueued, waitDone), elapsed_us(writeStart, writeDone));
  g_state.readback.Unmap();
}
} // namespace portmaster_fbdev
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
#ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
    BACKEND_OPENGL,
#endif
#ifdef DAWN_ENABLE_BACKEND_OPENGLES
    BACKEND_OPENGLES,
#endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();

#ifdef AURORA_ENABLE_GX
  gfx::initialize();

  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

#ifdef AURORA_ENABLE_GX
wgpu::TextureView g_currentView;
#endif

void shutdown() noexcept {
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
#ifdef AURORA_ENABLE_GX
  g_currentView = {};
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
  portmaster_fbdev::shutdown();
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  if (portmaster_fbdev::enabled()) {
    imgui::new_frame(window::get_window_size());
    if (!gfx::begin_frame()) {
      g_currentView = {};
      return false;
    }
    return true;
  }
  {
    window::SurfaceLock surfaceLock;
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
    wgpu::SurfaceTexture surfaceTexture;
    g_surface.GetCurrentTexture(&surfaceTexture);
    switch (surfaceTexture.status) {
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
      g_currentView = surfaceTexture.texture.CreateView();
      break;
    case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
      Log.warn("Surface texture acquisition timed out");
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
    case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
      Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceTexture.status));
      webgpu::refresh_surface(false);
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::Lost:
      Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceTexture.status));
      webgpu::release_surface();
    case wgpu::SurfaceGetCurrentTextureStatus::Error:
      Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceTexture.status));
      g_surface = {};
      return false;
    default:
      Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceTexture.status));
      return false;
    }
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    g_currentView = {};
    return false;
  }
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  const bool fbdevPresentThisFrame = portmaster_fbdev::should_present_frame();
  const auto frameStart = Clock::now();
  const auto drainStart = frameStart;
  gx::fifo::drain();
  const auto drainDone = Clock::now();
  const auto encoderDescriptor = wgpu::CommandEncoderDescriptor{
      .label = "Redraw encoder",
  };
  auto encoder = g_device.CreateCommandEncoder(&encoderDescriptor);
  const auto gfxEndStart = Clock::now();
  gfx::end_frame(encoder);
  const auto gfxEndDone = Clock::now();
  const auto renderStart = Clock::now();
  gfx::render(encoder);
  {
    window::SurfaceLock surfaceLock;
    if (window::is_presentable() && g_surface && g_currentView) {
      const auto& presentSource = webgpu::present_source();
      auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                         webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                         presentSource.size.width, presentSource.size.height);
      const auto& resampledSource = webgpu::resample_present_source(encoder, viewport);
      wgpu::BindGroup presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
    #if AURORA_ENABLE_RMLUI
      if (rmlui::is_initialized()) {
        const auto rmlOutput = rmlui::render(encoder, viewport, resampledSource);
        if (rmlOutput.texture != nullptr) {
          presentBindGroup = rmlOutput.copyBindGroup;
        }
      }
    #endif
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);

        pass.Draw(3);
        pass.End();
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass);
        pass.End();
      }
    } else if (!portmaster_fbdev::enabled()) {
      Log.info("Skipping present; window not presentable");
      webgpu::release_surface();
    }
    if (fbdevPresentThisFrame) {
      const auto& presentSource = webgpu::present_source();
      const webgpu::TextureWithSampler* readbackSource = &presentSource;
    #if AURORA_ENABLE_RMLUI
      auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                         webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                         presentSource.size.width, presentSource.size.height);
      if (rmlui::is_initialized()) {
        const auto rmlOutput = rmlui::render(encoder, viewport, presentSource);
        if (rmlOutput.texture != nullptr) {
          readbackSource = rmlOutput.texture;
        }
      }
    #endif
      portmaster_fbdev::enqueue_readback(encoder, *readbackSource);
    }
    const auto renderDone = Clock::now();
    const auto finishSubmitStart = Clock::now();
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
    const auto finishDone = Clock::now();
    g_queue.Submit(1, &buffer);
    const auto queueSubmitDone = Clock::now();
    gfx::after_submit();
    const auto submitDone = Clock::now();
    const auto fbdevStart = Clock::now();
    if (fbdevPresentThisFrame) {
      portmaster_fbdev::present_after_submit();
    }
    const auto fbdevDone = Clock::now();
    if (window::is_presentable() && g_surface) {
      auto presentStatus = g_surface.Present();
      if (presentStatus != wgpu::Status::Success) {
        Log.warn("Surface present failed: {}", static_cast<int>(presentStatus));
        webgpu::release_surface();
      }
    } else if (g_surface) {
      webgpu::release_surface();
    }
    g_currentView = {};
    note_portmaster_end_frame_timing(elapsed_us(drainStart, drainDone), elapsed_us(gfxEndStart, gfxEndDone),
                                     elapsed_us(renderStart, renderDone), elapsed_us(finishSubmitStart, finishDone),
                                     elapsed_us(finishDone, queueSubmitDone), elapsed_us(queueSubmitDone, submitDone),
                                     elapsed_us(finishSubmitStart, submitDone),
                                     elapsed_us(fbdevStart, fbdevDone), elapsed_us(frameStart, fbdevDone));
  }

  TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

  TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(gfx::g_stats.queuedPipelines));
  TracyPlot("aurora: createdPipelines", static_cast<int64_t>(gfx::g_stats.createdPipelines));
  TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
  TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
  TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
  TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
  TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
  TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(gfx::g_stats.lastStorageSize));
  TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(gfx::g_stats.lastTextureUploadSize));

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
void aurora_set_resampler(AuroraSampler sampler) {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::set_resampler(sampler);
#else
  (void)sampler;
#endif
}
