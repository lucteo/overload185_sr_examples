#include <exec/system_context.hpp>
#include <stdexec/execution.hpp>

int main() {
  stdexec::scheduler auto sched = exec::get_system_scheduler();
  stdexec::sender auto snd =
      stdexec::schedule(sched) | stdexec::then([] { printf("Hello, concurrency!\n"); });
  stdexec::sync_wait(std::move(snd));
}
