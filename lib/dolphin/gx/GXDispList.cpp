#include "gx.hpp"
#include "__gx.h"

#include <aurora/dl.hpp>
#include <dolphin/gx/GXGet.h>

#include "../../gx/command_processor.hpp"
#include "../../gx/fifo.hpp"

#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

static __GXData_struct sSavedGXData;

namespace {

struct PortmasterDlCacheKey {
  const void* data = nullptr;
  u32 size = 0;
  bool bigEndian = true;
  u64 dataHash = 0;
  u32 vcdLo = 0;
  u32 vcdHi = 0;
  u32 nrmState = 0;
  u32 vatA[GX_MAX_VTXFMT]{};
  u32 vatB[GX_MAX_VTXFMT]{};
  u32 vatC[GX_MAX_VTXFMT]{};

  bool operator==(const PortmasterDlCacheKey& rhs) const noexcept {
    return data == rhs.data && size == rhs.size && bigEndian == rhs.bigEndian && dataHash == rhs.dataHash &&
           vcdLo == rhs.vcdLo && vcdHi == rhs.vcdHi && nrmState == rhs.nrmState &&
           std::memcmp(vatA, rhs.vatA, sizeof(vatA)) == 0 && std::memcmp(vatB, rhs.vatB, sizeof(vatB)) == 0 &&
           std::memcmp(vatC, rhs.vatC, sizeof(vatC)) == 0;
  }
};

struct PortmasterDlCacheKeyHash {
  size_t operator()(const PortmasterDlCacheKey& key) const noexcept {
    u64 hash = key.dataHash ^ (reinterpret_cast<uintptr_t>(key.data) + 0x9e3779b97f4a7c15ull);
    hash ^= static_cast<u64>(key.size) << 32;
    hash ^= key.bigEndian ? 0xa0761d6478bd642full : 0xe7037ed1a0b428dbull;
    hash ^= static_cast<u64>(key.vcdLo) << 1;
    hash ^= static_cast<u64>(key.vcdHi) << 33;
    hash ^= static_cast<u64>(key.nrmState) << 17;
    for (u32 i = 0; i < GX_MAX_VTXFMT; ++i) {
      hash ^= static_cast<u64>(key.vatA[i]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
      hash ^= static_cast<u64>(key.vatB[i]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
      hash ^= static_cast<u64>(key.vatC[i]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    return static_cast<size_t>(hash);
  }
};

static std::unordered_map<PortmasterDlCacheKey, std::vector<u8>, PortmasterDlCacheKeyHash> sPortmasterDlCache;
static std::unordered_set<PortmasterDlCacheKey, PortmasterDlCacheKeyHash> sPortmasterDlOptimizeFailures;

static bool portmaster_dl_optimize_enabled() {
  static const bool enabled = [] {
    const char* disabled = std::getenv("DUSKLIGHT_PORTMASTER_DL_OPTIMIZE");
    if (disabled != nullptr && disabled[0] == '0') {
      return false;
    }
    return std::getenv("DUSKLIGHT_PORTMASTER_EGL_FBDEV_SURFACE") != nullptr ||
           std::getenv("DUSKLIGHT_PORTMASTER_SDL2SHIM_EGL_SURFACE") != nullptr ||
           std::getenv("DUSKLIGHT_PORTMASTER_FORCE_VERTEX_TEXTURE") != nullptr;
  }();
  return enabled;
}

static u64 portmaster_hash_display_list(const u8* data, u32 size) {
  u64 hash = 1469598103934665603ull;
  for (u32 i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

static bool portmaster_display_list_contains_indexed_draw(const u8* data, u32 size) {
  if (data == nullptr || size < 3) {
    return false;
  }
  for (u32 i = 0; i + 2 < size; ++i) {
    if (data[i] == GX_AURORA && data[i + 1] == 0x00 && data[i + 2] == GX_AURORA_DRAW_INDEXED) {
      return true;
    }
  }
  return false;
}

static PortmasterDlCacheKey portmaster_make_dl_cache_key(const void* data, u32 nbytes, bool bigEndian) {
  PortmasterDlCacheKey key;
  key.data = data;
  key.size = nbytes;
  key.bigEndian = bigEndian;
  key.dataHash = portmaster_hash_display_list(static_cast<const u8*>(data), nbytes);
  key.vcdLo = __gx->vcdLo;
  key.vcdHi = __gx->vcdHi;
  key.nrmState = static_cast<u32>(__gx->hasNrms) | (static_cast<u32>(__gx->hasBiNrms) << 8) |
                 (static_cast<u32>(__gx->nrmType) << 16);
  std::memcpy(key.vatA, __gx->vatA, sizeof(key.vatA));
  std::memcpy(key.vatB, __gx->vatB, sizeof(key.vatB));
  std::memcpy(key.vatC, __gx->vatC, sizeof(key.vatC));
  return key;
}

static const u8* portmaster_optimized_display_list(const void* data, u32 nbytes, bool bigEndian, u32& outSize) {
  outSize = nbytes;
  if (!portmaster_dl_optimize_enabled() || !bigEndian || data == nullptr || nbytes == 0) {
    return static_cast<const u8*>(data);
  }

  // Some J3D paths pre-optimize display lists once at resource load time.
  // Re-optimizing those lists each draw adds log noise and risks rejecting
  // already-packed indexed payloads that are ready for the command processor.
  if (portmaster_display_list_contains_indexed_draw(static_cast<const u8*>(data), nbytes)) {
    return static_cast<const u8*>(data);
  }

  const auto key = portmaster_make_dl_cache_key(data, nbytes, bigEndian);
  if (auto it = sPortmasterDlCache.find(key); it != sPortmasterDlCache.end()) {
    outSize = static_cast<u32>(it->second.size());
    return it->second.data();
  }
  if (sPortmasterDlOptimizeFailures.find(key) != sPortmasterDlOptimizeFailures.end()) {
    return static_cast<const u8*>(data);
  }

  GXVtxDescList desc[GX_VA_MAX_ATTR + 2]{};
  aurora::gx::dl::VtxFmtLists fmtLists{};
  GXVtxAttrFmtList fmtStorage[GX_MAX_VTXFMT][GX_VA_MAX_ATTR + 1]{};
  GXGetVtxDescv(desc);
  for (u32 fmt = 0; fmt < GX_MAX_VTXFMT; ++fmt) {
    GXGetVtxAttrFmtv(static_cast<GXVtxFmt>(fmt), fmtStorage[fmt]);
    fmtLists[fmt] = fmtStorage[fmt];
  }

  auto optimized = aurora::gx::dl::optimize(static_cast<const u8*>(data), nbytes, desc, &fmtLists);
  if (!optimized) {
    if (sPortmasterDlOptimizeFailures.size() > 4096) {
      sPortmasterDlOptimizeFailures.clear();
    }
    sPortmasterDlOptimizeFailures.insert(key);
    return static_cast<const u8*>(data);
  }

  if (sPortmasterDlCache.size() > 4096) {
    sPortmasterDlCache.clear();
    sPortmasterDlOptimizeFailures.clear();
  }
  auto [it, inserted] = sPortmasterDlCache.emplace(key, std::move(*optimized));
  if (!inserted) {
    it->second = std::move(*optimized);
  }
  outSize = static_cast<u32>(it->second.size());
  return it->second.data();
}

} // namespace

extern "C" {
void GXBeginDisplayList(void* list, u32 size) {
  CHECK(!aurora::gx::fifo::in_display_list(), "Display list began twice!");

  // Flush any pending dirty state before recording
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Save current shadow register state if requested
  if (__gx->dlSaveContext != 0) {
    std::memcpy(&sSavedGXData, __gx, sizeof(sSavedGXData));
  }

  __gx->inDispList = 1;

  // Redirect FIFO writes to the user-provided buffer
  aurora::gx::fifo::begin_display_list(static_cast<u8*>(list), size);
}

u32 GXEndDisplayList() {
  // Flush any pending dirty state into the display list
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // End FIFO redirection and get the byte count (ROUNDUP32)
  u32 bytesWritten = aurora::gx::fifo::end_display_list();

  // Restore saved shadow register state
  if (__gx->dlSaveContext != 0) {
    std::memcpy(__gx, &sSavedGXData, sizeof(*__gx));
  }

  __gx->inDispList = 0;

  return bytesWritten;
}

void GXCallDisplayList(const void* data, u32 nbytes) {
  // Flush any pending dirty state before calling
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Flush pending primitives
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }

  // Drain the internal FIFO so that any pending CP register writes
  // (VCD, VAT, etc.) are processed into g_gxState before the display
  // list's draw commands reference them.
  aurora::gx::fifo::drain();

  // Process the display list through the command processor
  u32 processedSize = nbytes;
  const u8* processedData = portmaster_optimized_display_list(data, nbytes, true, processedSize);
  aurora::gx::fifo::process(processedData, processedSize, true);
}

void GXCallDisplayListLE(const void* data, u32 nbytes) {
  // Flush any pending dirty state before calling
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Flush pending primitives
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }

  // Drain the internal FIFO so that any pending CP register writes
  // (VCD, VAT, etc.) are processed into g_gxState before the display
  // list's draw commands reference them.
  aurora::gx::fifo::drain();

  // Process the display list through the command processor (little-endian)
  u32 processedSize = nbytes;
  const u8* processedData = portmaster_optimized_display_list(data, nbytes, false, processedSize);
  aurora::gx::fifo::process(processedData, processedSize, false);
}
}
