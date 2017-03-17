#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctime>

#include <opencv2/opencv.hpp>

//创建目录
void MakeDir(const std::string& filePath);

//获取文件夹名
void GetDirList(const std::string& dirPath, std::vector<std::string>* dirList);

//获取文件名
void GetFileList(const std::string& filePath, std::vector<std::string>* fileList);

//生成n个互相不重复的随机数
std::vector<int> my_random(int number);

//生成libsvm中heart_scale格式的txt文件
void gen_txt_file(std::string llcDir, char* trainFile, char* testFile, int trNum);

#endif // UTILS_H_INCLUDED
