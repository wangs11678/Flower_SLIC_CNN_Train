#include "utils.h"

using namespace std;
using namespace cv;

void cnn_image_pretreatment(Mat &image, int imgSize)
{
	Mat img;
	resize(image, img, Size(imgSize, imgSize), 0, 0, CV_INTER_LINEAR);
	image = img;
}

//创建目录
void MakeDir(const string& filePath)
{
    DIR *dp;
	if((dp = opendir(filePath.c_str())) == NULL)
    {
        if(mkdir(filePath.c_str(), 0755) == -1)
            cout << filePath << " build failed!" << endl;
    }

    closedir(dp);
}

//获得文件夹名
void GetDirList(const string& dirPath, vector<string>* dirList)
{
	DIR *dp;
	struct dirent *dirp;

	if((dp = opendir(dirPath.c_str())) == NULL)
		cout << "Can't open " << dirPath << endl;

	while((dirp = readdir(dp)) != NULL)
    {
        if(dirp->d_type == DT_DIR && strcmp(".", dirp->d_name)
                    && strcmp("..", dirp->d_name))  // 只输出文件夹名
        {
            dirList->push_back(dirp->d_name);
        }
    }

	closedir(dp);
}

//获取文件名
void GetFileList(const string& filePath, vector<string>* fileList)
{
    DIR *dp;
	struct dirent *dirp;

	if((dp = opendir(filePath.c_str())) == NULL)
		cout << "Can't open " << filePath << endl;

	while((dirp = readdir(dp)) != NULL)
    {
        if(dirp->d_type == DT_REG)  // 只输出文件名
        {
            fileList->push_back(dirp->d_name);
        }
    }

	closedir(dp);
}

//生成n个互相不重复的随机数
vector<int> my_random(int number)
{
    vector<int> result;
    result.clear();
    result.reserve(number);
    srand((int)time(0));
    for (size_t i = 0; i < number; i++)
    {
        result.push_back(i);
    }
    int p1;
    int p2;
    int temp;
    int count = number;

    while (--number)
    {
        p1 = number;
        p2 = rand() % number;
        temp = result[p1];
        result[p1] = result[p2];
        result[p2] = temp;
    }
    return result;
}

//生成libsvm中heart_scale格式的txt文件
void gen_txt_file(string llcDir, char* trainFile, char* testFile, int trNum)
{
    cout << "Generate txt..." << endl;
    ofstream trFile(trainFile);
    ofstream tsFile(testFile);

    vector<string> categories;
    GetDirList(llcDir, &categories);

    Mat fc7;
    for (int i = 0; i != categories.size(); i++)
    {
        string currentCategory = llcDir + "/" + categories[i];

        vector<string> fileList;
        GetFileList(currentCategory, &fileList);

        /*my_random函数产生每类样本数的不重复随机数
         *（将每类样本次序打乱），存入vector, 然后从中
         *抽取前trNUM个数作为训练样本，剩余的作为测试样本
        */
        vector<int> randFileList;
        randFileList = my_random(fileList.size());

        for (int j = 0; j != trNum; j++)
        {
            trFile<<atoi(categories[i].c_str())<<" ";

            //string filePath = currentCategory + "/" + fileList[j];
            string filePath = currentCategory + "/" + fileList[randFileList[j]];

            FileStorage fs(filePath, FileStorage::READ);
            if (fs.isOpened())
            {
                fs ["fc7"] >> fc7;
                for(int k = 0; k < fc7.rows; k++)
                {
                    if(fc7.at<float>(k, 0) != 0)
                        trFile<<k+1<<":"<<fc7.at<float>(k, 0)<<" ";
                }
            }
            trFile<<"\n";
        }

        for (int j = trNum; j != fileList.size(); j++)
        {
            tsFile<<atoi(categories[i].c_str())<<" ";

            //string filePath = currentCategory + "/" + fileList[j];
            string filePath = currentCategory + "/" + fileList[randFileList[j]];

            FileStorage fs(filePath, FileStorage::READ);
            if (fs.isOpened())
            {
                fs ["fc7"] >> fc7;
                for(int k = 0; k < fc7.rows; k++)
                {
                    if(fc7.at<float>(k, 0) != 0)
                        tsFile<<k+1<<":"<<fc7.at<float>(k, 0)<<" ";
                }
            }
            tsFile<<"\n";
        }

    }
}
