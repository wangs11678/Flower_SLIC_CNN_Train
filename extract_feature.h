#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

std::vector<float> ExtractFeature(Mat FaceROI);//给一个图片 返回一个vector<float>容器

void Caffe_Predefine(std::string param_file, std::string pretrained_param_file);

void gen_alexnet_features(std::string imgDir, std::string featDir, int imgSize);
