/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include <algorithm>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <benchmark.h>

#include <SYCL/sycl.hpp>

enum class filter_type {
  identity,
  blur,
  edge,
};

template <typename T>
class image_ref {
 public:
  image_ref(T* imageData, int width, int height, int channels, int halo)
      : imageData_{imageData},
        width_{width},
        height_{height},
        channels_{channels},
        halo_{halo} {}

  ~image_ref() { delete[] imageData_; }

  T* data() const noexcept { return imageData_; }

  int width() const noexcept { return width_; }

  int height() const noexcept { return height_; }

  int channels() const noexcept { return channels_; }

  int halo() const noexcept { return halo_; }

  int count() const noexcept { return width_ * height_; }

  int size() const noexcept { return width_ * height_ * channels_; }

  int half_width() const noexcept { return width_ / 2; }

 private:
  T* imageData_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;
  int halo_ = 0;
};

image_ref<float> read_image(std::string imageFile, int halo) {
  int width = 0, height = 0, channels = 0;
  unsigned char* inputData =
      stbi_load(imageFile.c_str(), &width, &height, &channels, 4);
  assert(inputData != nullptr);

  int widthWithPadding = width + (halo * 2);
  int heightWithPadding = height + (halo * 2);

  int sizeWithPadding = (width + (halo * 2)) * (height + (halo * 2)) * channels;

  float* imageData = new float[sizeWithPadding];

  for (int i = 0; i < (heightWithPadding); ++i) {
    for (int j = 0; j < (widthWithPadding); ++j) {
      for (int c = 0; c < channels; ++c) {
        int srcI = i - halo;
        int srcJ = j - halo;

        srcJ = std::clamp(srcJ, 0, (width - 1));
        srcI = std::clamp(srcI, 0, (height - 1));

        int srcIndex = (srcI * width * channels) + (srcJ * channels) + c;
        int destIndex = (i * widthWithPadding * channels) + (j * channels) + c;

        imageData[destIndex] = static_cast<float>(inputData[srcIndex]);
      }
    }
  }

  stbi_image_free(inputData);

  return image_ref<float>{imageData, width, height, channels, halo};
}

image_ref<float> allocate_image(int width, int height, int channels) {
  float* imageData = new float[width * height * channels];

  return image_ref<float>{imageData, width, height, channels, 0};
}

template <typename T>
void write_image(const image_ref<T>& image, std::string imageFile) {
  unsigned char* rawOutputData = new unsigned char[image.size()];
  for (int i = 0; i < image.size(); ++i) {
    rawOutputData[i] = static_cast<unsigned char>(image.data()[i]);
  }

  stbi_write_png(imageFile.c_str(), image.width(), image.height(),
                 image.channels(), rawOutputData, 0);
}

image_ref<float> generate_filter(filter_type filterType, int width) {
  int count = width * width;
  int size = count * 4;

  float* filterData = new float[size];

  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < width; ++i) {
      auto index = ((j * width * 4) + (i * 4));
      auto isCenter = (j == (width / 2) && i == (width / 2));
      switch (filterType) {
        case filter_type::identity:
          filterData[index + 0] = isCenter ? 1.0f : 0.0f;
          filterData[index + 1] = isCenter ? 1.0f : 0.0f;
          filterData[index + 2] = isCenter ? 1.0f : 0.0f;
          filterData[index + 3] = isCenter ? 1.0f : 0.0f;
          break;
        case filter_type::blur:
          filterData[index + 0] = 1.0f / static_cast<float>(count);
          filterData[index + 1] = 1.0f / static_cast<float>(count);
          filterData[index + 2] = 1.0f / static_cast<float>(count);
          filterData[index + 3] = isCenter ? 1.0f : 0.0f;
          break;
        case filter_type::edge:
          filterData[index + 0] =
              isCenter ? static_cast<float>(count - 1) : -1.0f;
          filterData[index + 1] =
              isCenter ? static_cast<float>(count - 1) : -1.0f;
          filterData[index + 2] =
              isCenter ? static_cast<float>(count - 1) : -1.0f;
          filterData[index + 3] = isCenter ? 1.0f : 0.0f;
          break;
      }
    }
  }

  return image_ref<float>{filterData, width, width, 4, 0};
}

class image_convolution;

inline constexpr filter_type filterType = filter_type::blur;
inline constexpr int filterWidth = 11;
inline constexpr int halo = filterWidth / 2;

TEST_CASE("image_convolution", "image_convolution_reference") {
  const char* inputImageFile =
      "C:/Work/repos/syclacademy/Code_Exercises/"
      "Exercise_15_Image_Convolution/"
      "dogs.png";
  const char* outputImageFile =
      "C:/Work/repos/syclacademy/Code_Exercises/"
      "Exercise_15_Image_Convolution/"
      "dogs_blurred.png";

  auto inputImage = read_image(inputImageFile, halo);

  auto outputImage = allocate_image(inputImage.width(), inputImage.height(),
                                    inputImage.channels());

  auto filter = generate_filter(filter_type::blur, filterWidth);

  try {
    sycl::queue myQueue{sycl::gpu_selector_v,
                        [](sycl::exception_list exceptionList) {
                          for (auto e : exceptionList) {
                            std::rethrow_exception(e);
                          }
                        }};

    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    auto inputImgWidth = inputImage.width();
    auto inputImgHeight = inputImage.height();
    auto channels = inputImage.channels();
    auto filterWidth = filter.width();
    auto halo = filter.half_width();

    auto globalRange = sycl::range(inputImgWidth, inputImgHeight);
    auto localRange = sycl::range(32, 1);
    auto ndRange = sycl::nd_range(globalRange, localRange);

    auto inBufRange = (inputImgWidth + (halo * 2)) * sycl::range(1, channels);
    auto outBufRange = inputImgHeight * sycl::range(1, channels);
    auto filterRange = filterWidth * sycl::range(1, channels);

    {
      auto inBuf = sycl::buffer{inputImage.data(), inBufRange};
      auto outBuf = sycl::buffer<float, 2>{outBufRange};
      auto filterBuf = sycl::buffer{filter.data(), filterRange};
      outBuf.set_final_data(outputImage.data());

      auto inBufVec = inBuf.reinterpret<sycl::float4>(inBufRange /
                                                      sycl::range(1, channels));
      auto outBufVec = outBuf.reinterpret<sycl::float4>(
          outBufRange / sycl::range(1, channels));
      auto filterBufVec = filterBuf.reinterpret<sycl::float4>(
          filterRange / sycl::range(1, channels));

      util::benchmark(
          [&]() {
            myQueue.submit([&](sycl::handler& cgh) {
              auto inputAcc =
                  inBufVec.get_access<sycl::access::mode::read>(cgh);
              auto outputAcc =
                  outBufVec.get_access<sycl::access::mode::write>(cgh);
              auto filterAcc =
                  filterBufVec.get_access<sycl::access::mode::read>(cgh);

              cgh.parallel_for<image_convolution>(
                  ndRange, [=](sycl::nd_item<2> item) {
                    auto globalId = item.get_global_id();
                    globalId = sycl::id{globalId[1], globalId[0]};

                    auto haloOffset = sycl::id(halo, halo);
                    auto src = (globalId + haloOffset);
                    auto dest = globalId;

                    auto sum = sycl::float4{0.0f, 0.0f, 0.0f, 0.0f};

                    for (int r = 0; r < filterWidth; ++r) {
                      for (int c = 0; c < filterWidth; ++c) {
                        auto srcOffset = sycl::id(src[0] + (r - halo),
                                                  src[1] + ((c - halo)));
                        auto filterOffset = sycl::id(r, c);

                        sum += inputAcc[srcOffset] * filterAcc[filterOffset];
                      }
                    }

                    outputAcc[dest] = sum;
                  });
            });

            myQueue.wait_and_throw();
          },
          100, "image convolution (vectorized)");
    }
  } catch (sycl::exception e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  write_image(outputImage, outputImageFile);

  REQUIRE(true);
}
