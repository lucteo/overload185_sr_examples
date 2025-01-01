#include <opencv2/imgcodecs.hpp>
#include <opencv2/xphoto.hpp>

#include <exec/async_scope.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/system_context.hpp>
#include <exec/task.hpp>
#include <stdexec/execution.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

cv::Mat tr_apply_mask(const cv::Mat& img_main, const cv::Mat& img_mask) {
  cv::Mat res;
  cv::bitwise_and(img_main, img_main, res, img_mask);
  return res;
}

cv::Mat tr_blur(const cv::Mat& src, int size) {
  cv::Mat res;
  cv::GaussianBlur(src, res, cv::Size(size, size), 0, 0, cv::BORDER_DEFAULT);
  return res;
}
cv::Mat tr_to_grayscale(const cv::Mat& src) {
  cv::Mat res;
  cv::cvtColor(src, res, cv::COLOR_BGR2GRAY);
  return res;
}
cv::Mat tr_adaptthresh(const cv::Mat& img, int block_size, int diff) {
  cv::Mat res;
  cv::adaptiveThreshold(img, res, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, block_size,
                        diff);
  return res;
}
cv::Mat tr_reducecolors(const cv::Mat& img, int num_colors) {
  auto size = img.rows * img.cols;
  cv::Mat data = img.reshape(1, size);
  data.convertTo(data, CV_32F);
  auto criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
  cv::Mat labels;
  cv::Mat1f colors;
  cv::kmeans(data, num_colors, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, colors);
  for (unsigned int i = 0; i < size; i++) {
    data.at<float>(i, 0) = colors(labels.at<int>(i), 0);
    data.at<float>(i, 1) = colors(labels.at<int>(i), 1);
    data.at<float>(i, 2) = colors(labels.at<int>(i), 2);
  }
  cv::Mat res = data.reshape(3, img.rows);
  res.convertTo(res, CV_8U);
  return res;
}
cv::Mat tr_oilpainting(const cv::Mat& img, int size, int dyn_ratio) {
  cv::Mat res;
  cv::xphoto::oilPainting(img, res, size, dyn_ratio, cv::COLOR_BGR2Lab);
  return res;
}

auto error_to_exception() {
  return stdexec::let_error([](auto e) {
    if constexpr (std::same_as<decltype((e)), std::exception_ptr>)
      return stdexec::just_error(e);
    else
      return stdexec::just_error(std::make_exception_ptr(std::runtime_error("other error")));
  });
}

auto tr_cartoonify(const cv::Mat& src, int blur_size, int num_colors, int block_size, int diff) {
  auto sched = exec::get_system_scheduler();
  stdexec::sender auto snd =                 //
      stdexec::when_all(                     //
          stdexec::transfer_just(sched, src) //
              | error_to_exception()         //
              | stdexec::then([=](const cv::Mat& src) {
                  auto blurred = tr_blur(src, blur_size);
                  auto gray = tr_to_grayscale(blurred);
                  auto edges = tr_adaptthresh(gray, block_size, diff);
                  return edges;
                }),
          stdexec::transfer_just(sched, src)            //
              | error_to_exception()                    //
              | stdexec::then([=](const cv::Mat& src) { //
                  return tr_reducecolors(src, num_colors);
                })                                                              //
          )                                                                     //
      | stdexec::then([](const cv::Mat& edges, const cv::Mat& reduced_colors) { //
          return tr_apply_mask(reduced_colors, edges);
        }) //
      ;
  return snd;
}

std::vector<std::byte> read_file(const fs::directory_entry& file) {
  std::vector<std::byte> res{file.file_size()};
  std::ifstream in(file.path(), std::ios::binary);
  in.read(reinterpret_cast<char*>(res.data()), res.size());
  return res;
}

void write_file(const char* filename, const std::vector<unsigned char>& data) {
  std::ofstream out(filename);
  out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

exec::task<int> process_files(const char* in_folder_name, const char* out_folder_name,
                              int blur_size, int num_colors, int block_size, int diff) {
  exec::async_scope scope;
  // BUG: `io_pool` gets destroyed, but we still use it at the end of the coroutine
  exec::static_thread_pool io_pool(1);
  auto io_sched = io_pool.get_scheduler();
  auto cpu_sched = exec::get_system_scheduler();

  int processed = 0;
  for (const auto& entry : fs::directory_iterator(in_folder_name)) {
    auto extension = entry.path().extension();
    if (!entry.is_regular_file() || (extension != ".jpg") && (extension != ".jpeg"))
      continue;
    auto in_filename = entry.path().string();
    auto out_filename = (fs::path(out_folder_name) / entry.path().filename()).string();
    printf("Processing %s\n", in_filename.c_str());

    auto file_content =
        co_await (stdexec::schedule(io_sched) | stdexec::then([=] { return read_file(entry); }));

    stdexec::sender auto work =
        stdexec::transfer_just(cpu_sched, cv::_InputArray::rawIn(file_content)) //
        | error_to_exception()                                                  //
        | stdexec::then([=](cv::InputArray file_content) -> cv::Mat {
            return cv::imdecode(file_content, cv::IMREAD_COLOR);
          }) //
        | stdexec::let_value([=](const cv::Mat& img) {
            return tr_cartoonify(img, blur_size, num_colors, block_size, diff);
          }) //
        | stdexec::then([=](const cv::Mat& img) {
            std::vector<unsigned char> out_image_content;
            if (!cv::imencode(extension, img, out_image_content)) {
              throw std::runtime_error("cannot encode image");
            }
            return out_image_content;
          })                              //
        | stdexec::continues_on(io_sched) //
        | stdexec::then([=](const std::vector<unsigned char>& bytes) {
            write_file(out_filename.c_str(), bytes);
          })                                                                   //
        | stdexec::then([=] { printf("Written %s\n", out_filename.c_str()); }) //
        | stdexec::then([&] { processed++; });

    scope.spawn(std::move(work));
  }
  co_await scope.on_empty();
  co_return processed;
}

int main(int argc, char** argv) {
  int blur_size = 3;
  int num_colors = 5;
  int block_size = 5;
  int diff = 5;
  auto everything = process_files("data", "out", blur_size, num_colors, block_size, diff);
  auto [processed] = stdexec::sync_wait(std::move(everything)).value();
  printf("Processed images: %d\n", processed);
  return 0;
}