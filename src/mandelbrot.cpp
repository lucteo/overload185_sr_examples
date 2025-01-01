#include <chrono>
#include <complex>
#include <concepts>
#include <thread>
#define GL_SILENCE_DEPRECATION 1
#include <glut.h>

#include <exec/async_scope.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/system_context.hpp>
#include <stdexec/execution.hpp>

using namespace std::chrono_literals;

int mandelbrot_core(std::complex<double> c, int depth) {
  int count = 0;
  std::complex<double> z = 0;
  for (int i = 0; i < depth; i++) {
    if (abs(z) >= 2.0)
      break;
    z = z * z + c;
    count++;
  }
  return count;
}

template <typename F>
void serial_mandelbrot(int* vals, int max_x, int max_y, int depth, F&& transform) {
  for (int y = 0; y < max_y; y++) {
    for (int x = 0; x < max_x; x++) {
      vals[y * max_x + x] = mandelbrot_core(transform(x, y), depth);
    }
  }
}

template <typename F>
void mandelbrot_concurrent(int* vals, int max_x, int max_y, int depth, F&& transform) {
  auto sched = exec::get_system_scheduler();

  auto snd = stdexec::schedule(sched) | stdexec::bulk(max_y, [=](int y) {
               for (int x = 0; x < max_x; x++) {
                 vals[y * max_x + x] = mandelbrot_core(transform(x, y), depth);
               }
             });

  stdexec::sync_wait(std::move(snd));
}

std::complex<double> pixel_to_complex(int x, int y, int max_x, int max_y, double offset_x = 0.0,
                                      double offset_y = 0.0, double scale = 1.0) {
  double x0 = offset_x + (x - max_x / 2) * 4.0 / max_x / scale;
  double y0 = offset_y + (y - max_y / 2) * 4.0 / max_y / scale;
  return std::complex<double>(x0, y0);
}

static constexpr int max_x = 1024;
static constexpr int max_y = 1080;
static constexpr int depth = 1000;

/// Called each frame to calculate and render the Mandelbrot set.
void display() {
  auto t0 = std::chrono::high_resolution_clock::now();

  static std::vector<int> depths(max_x * max_y, 0);

  static unsigned char displayFrame[max_y][max_x][3];

  static double scale = 1.0;
  static bool scale_up = true;

  // Compute the Mandelbrot set, at different scales.
  auto transform = [](int x, int y) -> std::complex<double> {
    return pixel_to_complex(x, y, max_x, max_y, -1.4011, 0.0, scale);
  };
  // serial_mandelbrot(depths.data(), max_x, max_y, depth, transform);
  mandelbrot_concurrent(depths.data(), max_x, max_y, depth, transform);
  constexpr double scale_factor = 21.78 / 20.0;
  if (scale_up) {
    scale *= scale_factor;
    scale_up = scale < 65536.0;
  } else {
    scale /= scale_factor;
    scale_up = scale < 1.0;
  }

  // Convert the depths to colors.
  for (int ys = 0; ys < max_y; ys++) {
    for (int xs = 0; xs < max_x; xs++) {
      double depth = depths[ys * max_x + xs];

      displayFrame[ys][xs][0] = depth * 2;
      displayFrame[ys][xs][1] = depth * 15;
      displayFrame[ys][xs][2] = depth * 30;
    }
  }

  // Do the drawing.
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawPixels(max_x, max_y, GL_BGR_EXT, GL_UNSIGNED_BYTE, displayFrame);
  glFinish();
  glutSwapBuffers();
  glutPostRedisplay();

  auto t1 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  printf("\rFrame time: %dms", int(dt));
  fflush(stdout);
}

/// Handle keyboard input.
void handle_keyboard_input(unsigned char Key, int x, int y) {
  if (Key == 27) {
    exit(0);
  } else if (Key == 'f') {
    glutFullScreen();
  }
}

int main(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

  glutInitWindowPosition(380, 0);
  glutInitWindowSize(max_x, max_y);

  glutCreateWindow("Main");

  glutDisplayFunc(display);
  glutKeyboardFunc(handle_keyboard_input);

  glutMainLoop();
}