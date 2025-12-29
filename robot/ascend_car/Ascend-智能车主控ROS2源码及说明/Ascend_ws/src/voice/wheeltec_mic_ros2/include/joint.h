#ifndef __JOINT_H__
#define __JOINT_H__

#include <iostream>
#include <cstring>

using namespace std;

char *OTHER;
char *WHOLE;
std::string head = "aplay -D plughw:CARD=Device,DEV=0 ";

std::string audio_path = "/home/wheeltec/catkin_ws/src/wheeltec_mic_ros2";

//字符串与字符数组拼接
char *join(std::string b, char *s2)
{
	char s1[600] = "";
	try
	{
		strcpy(s1, b.c_str());
	}
	catch (...)
	{
		cout << ">>>>>join拷贝失败" << endl;
	}
	char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	if (result == NULL)
		exit(1);

	try
	{
		strcpy(result, s1);
		strcat(result, s2);
	}
	catch (...)
	{
		cout << ">>>>>join拷贝失败" << endl;
	}
	return result;
}

//判断是否是整数
int isnumber(char *a, int count_need)
{
	int len = strlen(a);
	if (len > count_need)
	{
		return -1;
	}
	int j = 0;
	for (int i = 0; i < len; i++)
	{
		if (a[i] <= 57 && a[i] >= 48)
		{
			j++;
		}
	}
	if (j == len)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

#endif