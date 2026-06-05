#include "BackendBinding.hpp"

#include <cstdlib>
#include <cstdint>
#include <fcntl.h>
#include <linux/fb.h>
#include <memory>
#include <sys/ioctl.h>
#include <unistd.h>

#if !defined(SDL_PLATFORM_MACOS) && !defined(SDL_PLATFORM_IOS) && !defined(SDL_PLATFORM_TVOS)
#include <SDL3/SDL_video.h>
#endif

namespace aurora::webgpu::utils {
std::shared_ptr<wgpu::ChainedStruct> SetupWindowAndGetSurfaceDescriptorCocoa(SDL_Window* window);

namespace {
#if defined(SDL_PLATFORM_LINUX)
struct PortmasterFbdevWindow {
  unsigned short width;
  unsigned short height;
};

PortmasterFbdevWindow g_portmasterFbdevWindow{};

bool init_portmaster_fbdev_window() {
  int fd = open("/dev/fb0", O_RDWR, 0);
  fb_var_screeninfo vinfo{};
  if (fd < 0 || ioctl(fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
    if (fd >= 0) {
      close(fd);
    }
    return false;
  }
  close(fd);
  g_portmasterFbdevWindow.width = static_cast<unsigned short>(vinfo.xres);
  g_portmasterFbdevWindow.height = static_cast<unsigned short>(vinfo.yres);
  return true;
}
#endif
} // namespace

std::shared_ptr<wgpu::ChainedStruct> SetupWindowAndGetSurfaceDescriptor(SDL_Window* window) {
#if defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_TVOS)
  return SetupWindowAndGetSurfaceDescriptorCocoa(window);
#else
  const auto props = SDL_GetWindowProperties(window);
#if defined(SDL_PLATFORM_ANDROID)
  std::shared_ptr<wgpu::SurfaceSourceAndroidNativeWindow> desc =
      std::make_shared<wgpu::SurfaceSourceAndroidNativeWindow>();
  desc->window = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr);
  return std::move(desc);
#elif defined(SDL_PLATFORM_WIN32)
  std::shared_ptr<wgpu::SurfaceSourceWindowsHWND> desc = std::make_shared<wgpu::SurfaceSourceWindowsHWND>();
  desc->hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
  desc->hinstance = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
  return std::move(desc);
#elif defined(SDL_PLATFORM_LINUX)
  if (std::getenv("DUSKLIGHT_PORTMASTER_EGL_FBDEV_SURFACE") != nullptr && init_portmaster_fbdev_window()) {
    std::shared_ptr<wgpu::SurfaceSourceXlibWindow> desc = std::make_shared<wgpu::SurfaceSourceXlibWindow>();
    desc->display = nullptr;
    desc->window = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&g_portmasterFbdevWindow));
    return std::move(desc);
  }
  const char* driver = SDL_GetCurrentVideoDriver();
  if (SDL_strcmp(driver, "wayland") == 0) {
    std::shared_ptr<wgpu::SurfaceSourceWaylandSurface> desc = std::make_shared<wgpu::SurfaceSourceWaylandSurface>();
    desc->display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
    desc->surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    return std::move(desc);
  }
  if (SDL_strcmp(driver, "x11") == 0) {
    std::shared_ptr<wgpu::SurfaceSourceXlibWindow> desc = std::make_shared<wgpu::SurfaceSourceXlibWindow>();
    desc->display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
    desc->window = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
    return std::move(desc);
  }
#endif
  return nullptr;
#endif
}

} // namespace aurora::webgpu::utils
