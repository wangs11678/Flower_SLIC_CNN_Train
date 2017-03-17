#define USE_OPENCV //必须放在第一行

#include "extract_feature.h"
#include "caffe_net_memorylayer.h"
#include "utils.h"

namespace caffe
{
    extern INSTANTIATE_CLASS(InputLayer);
    extern INSTANTIATE_CLASS(InnerProductLayer);
    extern INSTANTIATE_CLASS(DropoutLayer);
    extern INSTANTIATE_CLASS(ConvolutionLayer);
    //REGISTER_LAYER_CLASS(Convolution);
    extern INSTANTIATE_CLASS(ReLULayer);
    //REGISTER_LAYER_CLASS(ReLU);
    extern INSTANTIATE_CLASS(PoolingLayer);
    //REGISTER_LAYER_CLASS(Pooling);
    extern INSTANTIATE_CLASS(LRNLayer);
    //REGISTER_LAYER_CLASS(LRN);
    extern INSTANTIATE_CLASS(SoftmaxLayer);
    //REGISTER_LAYER_CLASS(Softmax);
    extern INSTANTIATE_CLASS(MemoryDataLayer);
}

template <typename Dtype>
caffe::Net<Dtype>* Net_Init_Load(std::string param_file, std::string pretrained_param_file, caffe::Phase phase)
{
    caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, phase));
    net->CopyTrainedLayersFrom(pretrained_param_file);
    return net;
}

void Caffe_Predefine(std::string param_file, std::string pretrained_param_file)//when our code begining run must add it
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    net = Net_Init_Load<float>(param_file, pretrained_param_file, caffe::TEST);
    memory_layer = (caffe::MemoryDataLayer<float> *)net->layers()[0].get();
}

std::vector<float> ExtractFeature(Mat FaceROI)
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    std::vector<Mat> test;
    std::vector<int> testLabel;
    std::vector<float> test_vector;
    test.push_back(FaceROI);
    testLabel.push_back(0);
    memory_layer->AddMatVector(test, testLabel);// memory_layer and net , must be define be a global variable.
    test.clear(); 
    testLabel.clear();
    std::vector<caffe::Blob<float>*> input_vec;
    net->Forward(input_vec);
    boost::shared_ptr<caffe::Blob<float> > fc7 = net->blob_by_name("fc7");
    int test_num = 0;
    while (test_num < 4096)
    {
        test_vector.push_back(fc7->data_at(0, test_num++, 1, 1));
    }
    return test_vector;
}

void gen_alexnet_features(string imgDir, string featDir, int imgSize)
{
	vector<string> categories;
	GetDirList(imgDir, &categories);

	for (int i = 0; i != categories.size(); i++)
	{
		cout<<"AlexNet Feature: "<<categories[i]<<"..."<<endl;
		string currentCategory = imgDir + "/" + categories[i];

		vector<string> fileList;
		GetFileList(currentCategory, &fileList);

		Mat fc7 = Mat(4096, 1, CV_32FC1);
		for (int j = 0; j != fileList.size(); j++)
		{
			MakeDir(featDir + "/" + categories[i]);
			string featFileName = featDir + "/" + categories[i] + "/" + fileList[j] + ".xml.gz";		
			
			FileStorage fs(featFileName, FileStorage::READ);
			if (fs.isOpened())
			{
				// already cached
				continue;
			}
			else
			{
				string filepath = currentCategory + "/" + fileList[j];
				Mat img = imread(filepath);
				cnn_image_pretreatment(img, 256);
				
				vector<float> out = ExtractFeature(img);
				int k = 0;
				while (k < out.size())
				{
				    fc7.at<float>(k, 0) = out[k++];
				}
				
				if (img.empty())
				{
					continue; // maybe not an image file
				}

				fs.open(featFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "fc7" << fc7;
				}
			}
		}
	}
}
