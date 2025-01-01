#include <chrono>
#include <cstdlib>

#include <exec/async_scope.hpp>
#include <exec/system_context.hpp>
#include <stdexec/execution.hpp>

using namespace std::chrono_literals;

static constexpr size_t size_threshold = 500;

template <typename T>
inline T med3(T v1, T v2, T v3) {
  return v1 < v2 ? (v2 < v3 ? v2 : (v1 < v3 ? v3 : v1)) : (v3 < v2 ? v2 : (v1 < v3 ? v3 : v1));
}

template <std::random_access_iterator It>
inline int median9(It it, int n) {
  assert(n >= 8);
  int stride = n / 8;
  int m1 = med3(*it, it[stride], it[stride * 2]);
  int m2 = med3(it[stride * 3], it[stride * 4], it[stride * 5]);
  int m3 = med3(it[stride * 6], it[stride * 7], it[n - 1]);
  return med3(m1, m2, m3);
}

template <std::random_access_iterator It>
std::pair<It, It> sort_partition(It first, It last) {
  auto n = static_cast<int>(std::distance(first, last));
  auto pivot = median9(first, n);
  auto mid1 = std::partition(first, last, [=](const auto& val) { return val < pivot; });
  auto mid2 = std::partition(first, last, [=](const auto& val) { return !(pivot < val); });
  return {mid1, mid2};
}

template <std::random_access_iterator It>
void serial_sort(It first, It last) {
  auto size = std::distance(first, last);
  if (size_t(size) < size_threshold) {
    // Use serial sort under a certain threshold.
    std::sort(first, last);
  } else {
    // Partition the data, such as elements [0, mid1) < [mid1, mid2) <= [mid2, n).
    // Elements in [mid1, mid2) are equal to the pivot.
    auto p = sort_partition(first, last);
    auto mid1 = p.first;
    auto mid2 = p.second;

    serial_sort(first, mid1);
    serial_sort(mid2, last);
  }
}

template <std::random_access_iterator It>
void concurrent_sort_impl(It first, It last, exec::async_scope& scope) {
  auto size = std::distance(first, last);
  if (size_t(size) < size_threshold) {
    // Use serial sort under a certain threshold.
    std::sort(first, last);
  } else {
    // Partition the data, such as elements [0, mid1) < [mid1, mid2) <= [mid2, n).
    // Elements in [mid1, mid2) are equal to the pivot.
    auto p = sort_partition(first, last);
    auto mid1 = p.first;
    auto mid2 = p.second;

    // Spawn work to sort the right-hand side.
    stdexec::sender auto snd                              //
        = stdexec::schedule(exec::get_system_scheduler()) //
          | stdexec::upon_error([](std::error_code ec) -> void {
              throw std::runtime_error("cannot start work");
            })                                                                      //
          | stdexec::then([=, &scope] { concurrent_sort_impl(mid2, last, scope); }) //
        ;
    scope.spawn(std::move(snd));
    // Execute the sorting on the left side, on the current thread.
    concurrent_sort_impl(first, mid1, scope);
  }
}

template <std::random_access_iterator It>
void concurrent_sort(It first, It last) {
  exec::async_scope scope;
  concurrent_sort_impl(first, last, scope);
  stdexec::sync_wait(scope.on_empty());
}

int main() {
  std::srand(0);
  std::vector<int> v;
  static constexpr int num_elem = 100'000'000;
  v.reserve(num_elem);
  for (int i = num_elem - 1; i >= 0; i--)
    v.push_back(rand());

  auto t0 = std::chrono::high_resolution_clock::now();
  concurrent_sort(v.begin(), v.end());
  // serial_sort(v.begin(), v.end());
  auto t1 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  if (std::is_sorted(v.begin(), v.end()))
    printf("Sorted\n");
  else
    printf("Not sorted\n");
  printf("Took %dms\n", int(dt));
}