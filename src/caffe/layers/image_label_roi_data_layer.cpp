#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/layers/image_label_roi_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


using std::string;
using std::vector;

namespace {

cv::Mat ReadImage(const std::string &filename) {
  int type_code;
  int shape_size;
  int count;
  std::vector<int> shape;
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp != NULL) << "Failed to open " << filename;
  count = fread((void*)(&type_code), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  count = fread((void*)(&shape_size), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  shape.resize(shape_size);
  count = fread((void*)shape.data(), sizeof(int), shape_size, fp);
  CHECK_EQ(count, shape_size);

  cv::Mat image(shape_size, shape.data(), type_code);
  count = fread((void*)image.data, image.elemSize1(),
                image.total(), fp);
  CHECK_EQ(count, image.total()) << "file: " << filename
  << " type: " << type_code << " shape size: " << shape_size
  << " shape: " << shape[0] << ' ' << shape[1] << ' '
  << shape[2];
  fclose(fp);

  return image;
}

template <typename Dtype>
void MirrorImage(cv::Mat &image) {
  CHECK_EQ(image.type(), CV_32F);
  int channels, height, width;
  if (image.dims == 2) {
    channels = 1;
    height = image.rows;
    width = image.cols;
  } else {
    channels = image.size[0];
    height = image.size[1];
    width = image.size[2];
  }
  Dtype *data = reinterpret_cast<Dtype*>(image.data);
  Dtype tmp;
  int idx0, idx1;
  for (int c = 0; c < channels; ++c) {
    Dtype *channel_data = data + c * height * width;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width / 2; ++w) {
        idx0 = h * width + w;
        idx1 = h * width + width - w - 1;
        tmp = channel_data[idx0];
        channel_data[idx0] = channel_data[idx1];
        channel_data[idx1] = tmp;
      }
    }
  }
}

cv::Mat PadImage(cv::Mat &image, int min_size, double value = -1) {
  if (image.rows >= min_size && image.cols >= min_size) {
    return image;
  }
  int top, bottom, left, right;
  top = bottom = left = right = 0;
  if (image.rows < min_size) {
    top = (min_size - image.rows) / 2;
    bottom = min_size - image.rows - top;
  }

  if (image.cols < min_size) {
    left = (min_size - image.cols) / 2;
    right = min_size - image.cols - left;
  }
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

cv::Mat ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h,
                          double value = -1) {
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

}

namespace caffe {

template <typename Dtype>
ImageLabelROIDataLayer<Dtype>::~ImageLabelROIDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageLabelROIDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  auto &data_param = this->layer_param_.bin_label_roi_data_param();

// Read the file with filenames and labels
  const string& image_list_path = data_param.bin_list_path();
  auto &label_slice = data_param.label_slice();
  label_margin_h_ = label_slice.offset(0);
  label_margin_w_ = label_slice.offset(1);
  LOG(INFO) << "Opening bin list " << image_list_path;
  std::ifstream infile(image_list_path.c_str());
  string filename;
  while (infile >> filename) {
    image_names_.push_back(filename);
  }
  infile.close();

  const string& label_list_path = data_param.label_list_path();
  LOG(INFO) << "Opening label list " << label_list_path;
  infile.open(label_list_path.c_str());
  while (infile >> filename) {
    label_names_.push_back(filename);
  }
  infile.close();

  const string& roi_list_path = data_param.roi_list_path();
  LOG(INFO) << "Opening roi list " << roi_list_path;
  infile.open(roi_list_path.c_str());
  while (infile >> filename) {
    roi_names_.push_back(filename);
  }
  infile.close();

  CHECK_EQ(roi_names_.size(), label_names_.size());
  CHECK_EQ(image_names_.size(), roi_names_.size());
  
  if (data_param.shuffle()) {
// randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << image_names_.size() << " images.";

  lines_id_ = 0;
  n_ = 0;

  cv::Mat image = ReadImageToCVMat(image_names_[0]);
  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }
  image = PadImage(image, crop_size);

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> data_shape = this->data_transformer_->InferBlobShape(image);
  const int batch_size = data_param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  data_shape[0] = batch_size;
  top[0]->Reshape(data_shape);

  vector<int> label_shape(2);
  label_shape[0] = batch_size;
  label_shape[1] = data_param.resolution();

  top[1]->Reshape(label_shape);

  vector<int> roi_shape(2);
  roi_shape[0] = batch_size;
  roi_shape[1] = 5;

  top[2]->Reshape(roi_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(data_shape);
    this->prefetch_[i].label_.Reshape(label_shape);
    this->prefetch_[i].roi_.Reshape(roi_shape);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();

  LOG(INFO) << "output label size: " << top[1]->num() << ","
  << top[1]->channels();

  LOG(INFO) << "output roi size: " << top[2]->num() << ","
  << top[2]->channels();

  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
void ImageLabelROIDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  vector<int> order(image_names_.size());
  for (int i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  shuffle(order.begin(), order.end(), prefetch_rng);
  vector<std::string> new_image_lines(image_names_.size());
  vector<std::string> new_label_lines(label_names_.size());
  vector<std::string> new_roi_lines(roi_names_.size());
  for (int i = 0; i < order.size(); ++i) {
    new_image_lines[i] = image_names_[order[i]];
    new_label_lines[i] = label_names_[order[i]];
    new_roi_lines[i] = roi_names_[order[i]];
  }
  swap(image_names_, new_image_lines);
  swap(label_names_, new_label_lines);
  swap(roi_names_, new_roi_lines);
}


template <typename Dtype>
int ImageLabelROIDataLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageLabelROIDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  auto &data_param = this->layer_param_.bin_label_roi_data_param();
  const int batch_size = data_param.batch_size();

  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }

  cv::Mat cv_img = ReadImageToCVMat(image_names_[lines_id_], true);
  cv_img = PadImage(cv_img, crop_size);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> data_shape = this->data_transformer_->InferBlobShape(cv_img);
  // Reshape prefetch_data according to the batch_size.
  data_shape[0] = batch_size;
  batch->data_.Reshape(data_shape);

  vector<int> label_shape(2);
  label_shape[0] = batch_size;
  label_shape[1] = data_param.resolution();
  batch->label_.Reshape(label_shape);

  vector<int> roi_shape(2);
  roi_shape[0] = batch_size;
  roi_shape[1] = 5;
  batch->roi_.Reshape(roi_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_roi = batch->roi_.mutable_cpu_data();

  // datum scales
  auto lines_size = image_names_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    bool do_mirror = data_param.mirror() && Rand(2);
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image

    int image_offset = batch->data_.offset(item_id);
    int label_offset = batch->label_.offset(item_id);
    int roi_offset = batch->roi_.offset(item_id);

    std::string image_path = image_names_[lines_id_];
    std::string label_path = label_names_[lines_id_];
    std::string roi_path = roi_names_[lines_id_];
    //std::cout << "Reading " << image_path << std::endl;
    cv::Mat cv_img =  ReadImageToCVMat(image_path);
    cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, -1);
    cv_img = PadImage(cv_img, crop_size, -1);
    // std::cout << "Reading " << label_path << std::endl;
    cv::Mat label = ReadImage(label_path);
    CHECK_GT(label.total(), 0);
    label.convertTo(label, CV_32F);

    cv::Mat roi = ReadImage(roi_path);
    CHECK_GT(roi.total(), 0);
    roi.convertTo(roi, CV_32F);

    cv::Mat label_row = label.row(n_);
    cv::Mat roi_row = roi.row(n_);

    if (do_mirror) {
      //MirrorImage<Dtype>(image);
      // MirrorLabel<Dtype>(label);
      //MirrorROI<Dtype>(label);
    }

    cv::Mat out_label(1, label_shape.data() + 1, cv::DataType<Dtype>::type,
                      prefetch_label + label_offset);
    CHECK_EQ(label_row.total(), out_label.total());
    CHECK_EQ(label_row.type(), out_label.type());
    // label_row.copyTo(out_label);
    for (int c = 0; c < 28; ++c) {
        out_label.at<float>(0, c) = label_row.at<float>(0, c);
    }

    cv::Mat out_roi(1, roi_shape.data() + 1, cv::DataType<Dtype>::type,
                      prefetch_roi + roi_offset);
    out_roi.at<float>(0, 0) = item_id;
    for (int c = 1; c < 5; ++c) {
        out_roi.at<float>(0, c) = roi_row.at<float>(0, c - 1);
    }
    CHECK_EQ(roi_row.type(), out_roi.type());

    //cv::Mat out_data(3, data_shape.data() + 1, cv::DataType<Dtype>::type,
    //                 prefetch_data + image_offset);
    //CHECK_EQ(cv_img.total(), out_data.total());
    //CHECK_EQ(cv_img.type(), out_data.type());
    //cv_img.copyTo(out_data);
    Dtype *image_data = prefetch_data + image_offset;
    for (int h = 0; h < cv_img.rows; ++h) {
      const uchar* ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_img.cols; ++w) {
        for (int c = 0; c < cv_img.channels(); ++c) {
          int top_index = (c * cv_img.rows + h) * cv_img.cols + w;
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          image_data[top_index] = pixel;
        }
      }
    }

    trans_time += timer.MicroSeconds();

    // std::cout << "Lines id: " <<  lines_id_ << " N: " << n_ << std::endl;
    n_++;
    if (n_ == label.rows) {
      n_ = 0;
      lines_id_++;
    }
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (data_param.shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageLabelROIDataLayer);
REGISTER_LAYER_CLASS(ImageLabelROIData);

}  // namespace caffe
