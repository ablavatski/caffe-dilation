#ifndef CAFFE_IMAGE_LABEL_ROI_DATA_LAYER_H
#define CAFFE_IMAGE_LABEL_ROI_DATA_LAYER_H

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class ImageLabelROIDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageLabelROIDataLayer(const LayerParameter &param)
      : BasePrefetchingDataLayer<Dtype>(param) { }

  virtual ~ImageLabelROIDataLayer();

  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "ImageLabelROIData"; }

  virtual inline int ExactNumBottomBlobs() const { return 0; }

  virtual inline int ExactNumTopBlobs() const { return -1; }

  virtual inline int MaxTopBlobs() const { return 3; }

  virtual inline int MinTopBlobs() const { return 3; }

  int Rand(int n);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;

  virtual void ShuffleImages();

  virtual void load_batch(Batch<Dtype> *batch);

  std::vector<std::string> image_names_;
  std::vector<std::string> label_names_;
  std::vector<std::string> roi_names_;
  int lines_id_;
  int n_;
  int label_margin_h_;
  int label_margin_w_;

  shared_ptr<Caffe::RNG> rng_;
};

} // namespace caffe

#endif //CAFFE_IMAGE_LABEL_ROI_DATA_LAYER_H
