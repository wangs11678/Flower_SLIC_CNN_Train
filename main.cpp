#include "extract_feature.h"
#include "utils.h"
#include "train.h"
#include "predict.h"

int main(int argc, char* argv[])
{
	MakeDir("result");
	MakeDir("result/alexnet");
	
	//初始化Caffe==================================================
	std::string caffe_root = "/home/wangs/caffe/";
	std::string param_file = caffe_root + "models/bvlc_alexnet/deploy_Test.prototxt";
	std::string pretrained_param_file = caffe_root + "models/bvlc_alexnet/bvlc_alexnet.caffemodel";
    Caffe_Predefine(param_file, pretrained_param_file);
	
	//提取特征=====================================================
	std::string imgDir = "images";
	std::string featDir = "result/alexnet";
	int imgSize = 256;
	gen_alexnet_features(imgDir, featDir, imgSize);
	
	//生成文档=====================================================
    int trNum = 1000;
    char trainFile[] = "result/train.txt";
    char testFile[] = "result/test.txt";
    gen_txt_file(featDir, trainFile, testFile, trNum);
    
    //训练数据=====================================================
    char modelFile[] = "result/model.txt";
    SVM_train(argc, argv, trainFile, modelFile);

    //测试数据=====================================================
    char resultFile[] = "result/result.txt";
    vector<int> labels;
    labels = SVM_predict(argc, argv, testFile, modelFile, resultFile);
    
    return 0;
}
