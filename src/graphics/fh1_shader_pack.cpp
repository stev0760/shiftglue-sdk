#include <rex/graphics/fh1_shader_pack.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <tuple>

#include <fmt/format.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <bcrypt.h>
#else
// The plugin has hidden visibility, so its own copy does not clash with the
// runtime's (xboxkrnl_crypt.cpp compiles the same file).
#include "thirdparty/crypto/sha256.cpp"
#include "thirdparty/crypto/sha256.h"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h>
#include <immintrin.h>
#define REX_FH1_PACK_SHA_NI 1
#endif
#endif

namespace rex::graphics {
namespace {

constexpr uint32_t kKnownFlags = 0xF;
constexpr uint32_t kKnownD3D12Features = Fh1ShaderPack::kD3D12FeatureSwitch;
constexpr size_t kMaximumEntries = 65'535;
constexpr size_t kMaximumBindings = 255;
constexpr size_t kMaximumBytecodeSize = 16 * 1024 * 1024;
// The Vulkan disc corpus is about 1.1 GB of SPIR-V at 2x.
constexpr size_t kMaximumPackSize = size_t(2) * 1024 * 1024 * 1024;

struct Header {
  char magic[8];
  uint32_t version;
  uint32_t header_size;
  uint32_t entry_size;
  uint32_t entry_count;
  uint64_t index_offset;
  uint64_t data_offset;
  uint64_t data_size;
  uint8_t content_sha256[32];
  uint32_t translator_version;
  uint32_t backend;
  uint32_t device_features;
  uint32_t flags;
  uint32_t draw_resolution_scale_x;
  uint32_t draw_resolution_scale_y;
  uint32_t reserved[2];
};
static_assert(sizeof(Header) == 112);

struct StoredEntry {
  uint32_t stage;
  // Equal to the header's backend: 1 DXBC, 2 SPIR-V.
  uint32_t bytecode_format;
  uint64_t guest_hash;
  uint64_t modification;
  uint64_t data_offset;
  uint64_t bytecode_size;
  uint32_t texture_binding_count;
  uint32_t sampler_binding_count;
  uint32_t used_texture_mask;
  uint32_t reserved;
  uint8_t bytecode_sha256[32];
};
static_assert(sizeof(StoredEntry) == 88);

struct StoredTextureBinding {
  uint32_t bindless_descriptor_index;
  uint32_t fetch_constant;
  uint32_t dimension;
  uint32_t is_signed;
};
static_assert(sizeof(StoredTextureBinding) == 16);

struct StoredSamplerBinding {
  uint32_t bindless_descriptor_index;
  uint32_t fetch_constant;
  uint32_t mag_filter;
  uint32_t min_filter;
  uint32_t mip_filter;
  uint32_t aniso_filter;
};
static_assert(sizeof(StoredSamplerBinding) == 24);

bool RangeValid(uint64_t offset, uint64_t size, size_t total_size) {
  return offset <= total_size && size <= total_size - offset;
}

#if defined(REX_FH1_PACK_SHA_NI)
// SHA-256 with the x86 SHA extensions, which CNG uses on Windows: the portable
// code took about three seconds a pass over a 1 GB Vulkan pack.
bool CpuHasShaExtensions() {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || !(ecx & bit_SSSE3) || !(ecx & bit_SSE4_1)) {
    return false;
  }
  return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & bit_SHA);
}

alignas(16) constexpr uint32_t kSha256RoundConstants[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4,
    0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE,
    0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F,
    0x4A7484AA, 0x5CB0A9DC, 0x76F988DA, 0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC,
    0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
    0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070, 0x19A4C116,
    0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7,
    0xC67178F2};

__attribute__((target("sha,ssse3,sse4.1"))) void Sha256BlocksWithShaExtensions(
    uint32_t state[8], const uint8_t* data, size_t block_count) {
  const __m128i byte_swap = _mm_set_epi64x(0x0C0D0E0F08090A0BLL, 0x0405060700010203LL);
  // The rounds keep the state as ABEF and CDGH.
  __m128i swapped = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(&state[0])), 0xB1);
  __m128i state1 = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(&state[4])), 0x1B);
  __m128i state0 = _mm_alignr_epi8(swapped, state1, 8);
  state1 = _mm_blend_epi16(state1, swapped, 0xF0);
  for (; block_count; --block_count, data += 64) {
    const __m128i state0_before = state0, state1_before = state1;
    // Four message words per group of four rounds, the last four groups'
    // in a ring: group n's words are made from groups n - 4 to n - 1.
    __m128i words[4];
    for (size_t i = 0; i < 4; ++i) {
      words[i] = _mm_shuffle_epi8(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * i)), byte_swap);
    }
    for (size_t n = 0; n < 16; ++n) {
      if (n >= 4) {
        __m128i next = _mm_sha256msg1_epu32(words[n & 3], words[(n + 1) & 3]);
        next = _mm_add_epi32(next, _mm_alignr_epi8(words[(n + 3) & 3], words[(n + 2) & 3], 4));
        words[n & 3] = _mm_sha256msg2_epu32(next, words[(n + 3) & 3]);
      }
      const __m128i message = _mm_add_epi32(
          words[n & 3],
          _mm_load_si128(reinterpret_cast<const __m128i*>(&kSha256RoundConstants[4 * n])));
      state1 = _mm_sha256rnds2_epu32(state1, state0, message);
      state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(message, 0x0E));
    }
    state0 = _mm_add_epi32(state0, state0_before);
    state1 = _mm_add_epi32(state1, state1_before);
  }
  swapped = _mm_shuffle_epi32(state0, 0x1B);
  state1 = _mm_shuffle_epi32(state1, 0xB1);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), _mm_blend_epi16(swapped, state1, 0xF0));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), _mm_alignr_epi8(state1, swapped, 8));
}

void Sha256WithShaExtensions(std::span<const uint8_t> bytes,
                             std::array<uint8_t, 32>& digest_out) {
  uint32_t state[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
  const size_t full_blocks = bytes.size() / 64;
  Sha256BlocksWithShaExtensions(state, bytes.data(), full_blocks);
  // The rest, the 0x80 marker and the big-endian bit count fill one or two
  // more blocks.
  uint8_t tail[128] = {};
  const size_t tail_size = bytes.size() % 64;
  std::memcpy(tail, bytes.data() + full_blocks * 64, tail_size);
  tail[tail_size] = 0x80;
  const size_t tail_blocks = tail_size < 56 ? 1 : 2;
  const uint64_t bit_count = uint64_t(bytes.size()) * 8;
  for (size_t i = 0; i < 8; ++i) {
    tail[tail_blocks * 64 - 1 - i] = uint8_t(bit_count >> (8 * i));
  }
  Sha256BlocksWithShaExtensions(state, tail, tail_blocks);
  for (size_t i = 0; i < 8; ++i) {
    digest_out[4 * i] = uint8_t(state[i] >> 24);
    digest_out[4 * i + 1] = uint8_t(state[i] >> 16);
    digest_out[4 * i + 2] = uint8_t(state[i] >> 8);
    digest_out[4 * i + 3] = uint8_t(state[i]);
  }
}
#endif

bool Sha256(std::span<const uint8_t> bytes, std::array<uint8_t, 32>& digest_out) {
#if defined(_WIN32)
  // CNG uses the CPU's SHA extensions; packs are hundreds of megabytes.
  if (bytes.size() > std::numeric_limits<ULONG>::max()) {
    return false;
  }
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  if (!BCRYPT_SUCCESS(
          BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0))) {
    return false;
  }
  const NTSTATUS status = BCryptHash(
      algorithm, nullptr, 0, const_cast<PUCHAR>(bytes.data()), static_cast<ULONG>(bytes.size()),
      digest_out.data(), static_cast<ULONG>(digest_out.size()));
  BCryptCloseAlgorithmProvider(algorithm, 0);
  return BCRYPT_SUCCESS(status);
#else
#if defined(REX_FH1_PACK_SHA_NI)
  static const bool sha_extensions = CpuHasShaExtensions();
  if (sha_extensions) {
    Sha256WithShaExtensions(bytes, digest_out);
    return true;
  }
#endif
  sha256::SHA256 hash;
  hash.add(bytes.data(), bytes.size());
  hash.getHash(digest_out.data());
  return true;
#endif
}

bool BytecodeMagicValid(Fh1ShaderPack::Backend backend, std::span<const uint8_t> bytecode) {
  if (bytecode.size() < 4) {
    return false;
  }
  if (backend == Fh1ShaderPack::Backend::kD3D12) {
    return !std::memcmp(bytecode.data(), "DXBC", 4);
  }
  // SPIR-V: whole 32-bit words, little-endian magic 0x07230203.
  static constexpr uint8_t kSpirvMagic[4] = {0x03, 0x02, 0x23, 0x07};
  return bytecode.size() % 4 == 0 && !std::memcmp(bytecode.data(), kSpirvMagic, 4);
}

bool SameLayout(const Fh1ShaderPack::Entry& left, const Fh1ShaderPack::Entry& right) {
  auto same_texture = [](const Fh1ShaderPack::TextureBinding& a,
                         const Fh1ShaderPack::TextureBinding& b) {
    return a.bindless_descriptor_index == b.bindless_descriptor_index &&
           a.fetch_constant == b.fetch_constant && a.dimension == b.dimension &&
           a.is_signed == b.is_signed;
  };
  auto same_sampler = [](const Fh1ShaderPack::SamplerBinding& a,
                         const Fh1ShaderPack::SamplerBinding& b) {
    return a.bindless_descriptor_index == b.bindless_descriptor_index &&
           a.fetch_constant == b.fetch_constant && a.mag_filter == b.mag_filter &&
           a.min_filter == b.min_filter && a.mip_filter == b.mip_filter &&
           a.aniso_filter == b.aniso_filter;
  };
  return left.used_texture_mask == right.used_texture_mask &&
         std::equal(left.texture_bindings.begin(), left.texture_bindings.end(),
                    right.texture_bindings.begin(), right.texture_bindings.end(),
                    same_texture) &&
         std::equal(left.sampler_bindings.begin(), left.sampler_bindings.end(),
                    right.sampler_bindings.begin(), right.sampler_bindings.end(),
                    same_sampler);
}

auto EntryKey(const Fh1ShaderPack::Entry& entry) {
  return std::tuple(static_cast<uint32_t>(entry.stage), entry.guest_hash, entry.modification);
}

}  // namespace

std::string Fh1ShaderPack::FileName(uint32_t title_id, const Config& config) {
  return fmt::format("{:08X}.fh1-native-v{}.{}.{:02X}.{:02X}.{}x{}.pnsp", title_id, kVersion,
                     config.backend == Backend::kVulkan ? "vulkan" : "d3d12",
                     config.device_features, config.flags, config.draw_resolution_scale_x,
                     config.draw_resolution_scale_y);
}

bool Fh1ShaderPack::Load(const std::filesystem::path& path, const Config& expected_config,
                         std::string* error_out) {
  Clear();
  auto fail = [&](const char* error) {
    if (error_out) {
      *error_out = error;
    }
    Clear();
    return false;
  };

  std::error_code size_error;
  const uintmax_t file_size = std::filesystem::file_size(path, size_error);
  if (size_error) {
    return fail("unavailable");
  }
  if (file_size < sizeof(Header) || file_size > kMaximumPackSize) {
    return fail("invalid_size");
  }
  auto mapping = rex::memory::MappedMemory::Open(path, rex::memory::MappedMemory::Mode::kRead);
  if (!mapping || mapping->size() != file_size) {
    return fail("read_failed");
  }
  const std::span<const uint8_t> data(mapping->data(), mapping->size());

  Header header;
  std::memcpy(&header, data.data(), sizeof(header));
  const Config actual_config{header.translator_version, Backend(header.backend),
                             header.device_features, header.flags,
                             header.draw_resolution_scale_x, header.draw_resolution_scale_y};
  constexpr char kMagic[8] = {'P', 'N', 'Y', 'N', 'S', 'H', 'P', 'K'};
  const bool backend_known = header.backend == uint32_t(Backend::kD3D12) ||
                             header.backend == uint32_t(Backend::kVulkan);
  const bool features_known = header.backend != uint32_t(Backend::kD3D12) ||
                              !(header.device_features & ~kKnownD3D12Features);
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) || header.version != kVersion ||
      header.header_size != sizeof(Header) || header.entry_size != sizeof(StoredEntry) ||
      !header.entry_count || header.entry_count > kMaximumEntries ||
      header.index_offset != sizeof(Header) || header.flags & ~kKnownFlags || !backend_known ||
      !features_known || actual_config != expected_config || header.reserved[0] ||
      header.reserved[1]) {
    return fail("incompatible_header");
  }
  const Backend backend = Backend(header.backend);
  const uint64_t index_size = uint64_t(header.entry_count) * sizeof(StoredEntry);
  if (!RangeValid(header.index_offset, index_size, data.size()) ||
      header.data_offset != header.index_offset + index_size ||
      !RangeValid(header.data_offset, header.data_size, data.size()) ||
      header.data_offset + header.data_size != data.size()) {
    return fail("invalid_ranges");
  }
  std::array<uint8_t, 32> content_digest;
  if (!Sha256(std::span(data).subspan(static_cast<size_t>(header.index_offset)), content_digest) ||
      std::memcmp(content_digest.data(), header.content_sha256, content_digest.size())) {
    return fail("content_hash_mismatch");
  }

  entries_.reserve(header.entry_count);
  uint64_t previous_payload_end = 0;
  for (uint32_t index = 0; index < header.entry_count; ++index) {
    StoredEntry stored;
    std::memcpy(&stored,
                data.data() + header.index_offset + uint64_t(index) * sizeof(StoredEntry),
                sizeof(stored));
    const bool geometry = stored.stage == uint32_t(Stage::kGeometry);
    if (stored.stage < uint32_t(Stage::kVertex) || stored.stage > uint32_t(Stage::kGeometry) ||
        stored.bytecode_format != header.backend || stored.reserved || !stored.bytecode_size ||
        stored.bytecode_size > kMaximumBytecodeSize ||
        stored.texture_binding_count > kMaximumBindings ||
        stored.sampler_binding_count > kMaximumBindings || stored.data_offset % 16 ||
        stored.data_offset < previous_payload_end ||
        (geometry && (stored.guest_hash || stored.texture_binding_count ||
                      stored.sampler_binding_count || stored.used_texture_mask))) {
      return fail("invalid_entry");
    }
    const uint64_t texture_bytes =
        uint64_t(stored.texture_binding_count) * sizeof(StoredTextureBinding);
    const uint64_t sampler_bytes =
        uint64_t(stored.sampler_binding_count) * sizeof(StoredSamplerBinding);
    const uint64_t entry_size = stored.bytecode_size + texture_bytes + sampler_bytes;
    if (!RangeValid(stored.data_offset, entry_size, static_cast<size_t>(header.data_size))) {
      return fail("entry_out_of_range");
    }
    for (uint64_t padding = previous_payload_end; padding < stored.data_offset; ++padding) {
      if (data[header.data_offset + padding]) {
        return fail("nonzero_padding");
      }
    }
    const uint8_t* entry_data = data.data() + header.data_offset + stored.data_offset;
    const std::span<const uint8_t> bytecode(entry_data, static_cast<size_t>(stored.bytecode_size));
    if (!BytecodeMagicValid(backend, bytecode)) {
      return fail("invalid_bytecode");
    }
    // The content hash above covers the bytecode too; hashing it again here
    // doubled the load of a 1 GB pack. The builder checks the entries' hashes.

    Entry entry;
    entry.stage = Stage(stored.stage);
    entry.guest_hash = stored.guest_hash;
    entry.modification = stored.modification;
    entry.bytecode = bytecode;
    const uint8_t* bindings_data = entry_data + stored.bytecode_size;
    uint32_t actual_texture_mask = 0;
    entry.texture_bindings.reserve(stored.texture_binding_count);
    for (uint32_t binding_index = 0; binding_index < stored.texture_binding_count;
         ++binding_index) {
      StoredTextureBinding binding;
      std::memcpy(&binding, bindings_data + binding_index * sizeof(binding), sizeof(binding));
      if (binding.fetch_constant >= 32 || binding.dimension > 3 || binding.is_signed > 1) {
        return fail("invalid_texture_binding");
      }
      actual_texture_mask |= 1u << binding.fetch_constant;
      entry.texture_bindings.push_back(
          {binding.bindless_descriptor_index, binding.fetch_constant,
           static_cast<xenos::FetchOpDimension>(binding.dimension), binding.is_signed != 0});
    }
    if (actual_texture_mask != stored.used_texture_mask) {
      return fail("texture_mask_mismatch");
    }
    bindings_data += texture_bytes;
    entry.sampler_bindings.reserve(stored.sampler_binding_count);
    for (uint32_t binding_index = 0; binding_index < stored.sampler_binding_count;
         ++binding_index) {
      StoredSamplerBinding binding;
      std::memcpy(&binding, bindings_data + binding_index * sizeof(binding), sizeof(binding));
      if (binding.fetch_constant >= 32 || binding.mag_filter > 3 || binding.min_filter > 3 ||
          binding.mip_filter > 3 || binding.aniso_filter > 7) {
        return fail("invalid_sampler_binding");
      }
      entry.sampler_bindings.push_back(
          {binding.bindless_descriptor_index, binding.fetch_constant,
           static_cast<xenos::TextureFilter>(binding.mag_filter),
           static_cast<xenos::TextureFilter>(binding.min_filter),
           static_cast<xenos::TextureFilter>(binding.mip_filter),
           static_cast<xenos::AnisoFilter>(binding.aniso_filter)});
    }
    entry.used_texture_mask = stored.used_texture_mask;
    if (!entries_.empty() && EntryKey(entry) <= EntryKey(entries_.back())) {
      return fail("unsorted_or_duplicate_identity");
    }
    if (!geometry && !entries_.empty() && entry.stage == entries_.back().stage &&
        entry.guest_hash == entries_.back().guest_hash && !SameLayout(entry, entries_.back())) {
      return fail("inconsistent_shader_layout");
    }
    entries_.push_back(std::move(entry));
    previous_payload_end = stored.data_offset + entry_size;
  }
  if (previous_payload_end != header.data_size) {
    return fail("trailing_payload");
  }
  mapping_ = std::move(mapping);
  return true;
}

const Fh1ShaderPack::Entry* Fh1ShaderPack::Find(Stage stage, uint64_t guest_hash,
                                                uint64_t modification) const {
  const auto key = std::tuple(static_cast<uint32_t>(stage), guest_hash, modification);
  const auto found = std::lower_bound(entries_.begin(), entries_.end(), key,
                                      [](const Entry& entry, const auto& wanted) {
                                        return EntryKey(entry) < wanted;
                                      });
  return found != entries_.end() && EntryKey(*found) == key ? &*found : nullptr;
}

}  // namespace rex::graphics
