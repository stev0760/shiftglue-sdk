/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2015 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <chrono>

#include <rex/cvar.h>
#include <rex/perf/counter.h>
#include <rex/platform.h>
#include <rex/thread.h>
#include <rex/thread/mutex.h>

// Windows' mutex already spins before it waits (an SRW lock underneath), so
// only the others spin here by default.
#if REX_PLATFORM_WIN32
#define REX_GLOBAL_LOCK_SPIN_US_DEFAULT 0
#else
#define REX_GLOBAL_LOCK_SPIN_US_DEFAULT 20
#endif

REXCVAR_DEFINE_INT32(global_lock_spin_us, REX_GLOBAL_LOCK_SPIN_US_DEFAULT, "CPU",
                     "Microseconds a thread spins for the global critical region before it "
                     "sleeps (0: sleep at once)")
    .range(0, 1000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::thread {

std::recursive_mutex& global_critical_region::mutex() {
  static std::recursive_mutex global_mutex;
  return global_mutex;
}

void global_critical_region::LockContended() {
  PROFILE_CRITICAL_REGION_CONTENTION();
  const auto start = std::chrono::steady_clock::now();
  const auto record_blocked = [start] {
    PERF_counter_add(kCriticalRegionBlockedNs,
                     std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count());
  };
  // glibc's mutex sleeps in the kernel at once, while most holders of this
  // lock let go within microseconds: the wake-up then costs more than the
  // wait, and the threads woken queue up behind one another. On Linux, threads
  // blocked here summed 14 ms a frame in free roam (113 ms in late frames),
  // against 0.34 ms on Windows. Spin briefly first, bounded in time so that
  // spinners never keep a preempted holder off a core for long.
  const auto spin = std::chrono::microseconds(REXCVAR_GET(global_lock_spin_us));
  if (spin.count() > 0) {
    const auto deadline = start + spin;
    do {
      for (int i = 0; i < 16; ++i) {
        SpinPause();
      }
      if (mutex().try_lock()) {
        record_blocked();
        return;
      }
    } while (std::chrono::steady_clock::now() < deadline);
  }
  mutex().lock();
  record_blocked();
}

}  // namespace rex::thread
