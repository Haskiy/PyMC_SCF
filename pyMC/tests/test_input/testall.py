#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2022-04-11 15:04:37
LastEditTime: 2022-04-13 12:54:11
LastEditors: Pu Zhichen
Description: 
FilePath: \pyMC\tests\test_input\testall.py

 May the force be with you!
'''

import os
import time
import numpy

path = os.getcwd()
all_files = os.listdir(path)
test_folders = [path + '/' + filename for filename in all_files if not os.path.isfile(filename)]

output_logs = []
test_name_list = []

for folder in test_folders:
    os.chdir(folder)
    test_files_tmp = os.listdir(folder)
    test_files = [filename for filename in test_files_tmp 
                    if filename.split('.')[-1] == 'py']
    for test in test_files:
        test_name_list.append(test)
        print("Runing " + test + " ...")
        # print('python ' + test)
        flag = True
        os.system('python3 ' + test + '> tmpout')
        with open('tmpout', 'r') as fn:
            lines = fn.readlines()
            for line in lines:
                if  line.upper().find('NEGATIVE') != -1:
                    flag = False
                    break
                else:
                    continue
        print("Finish " + test + " ...")
        output_logs.append(flag)
    all_files = os.listdir(folder)
    for afile in all_files:
        if afile in test_files:
            pass
        else:
            # print(afile)
            os.remove(afile) 
            
Ntotal_tests = int(len(output_logs))
Npositive = numpy.array(output_logs).sum()
Nnegative = Ntotal_tests-Npositive
idx = numpy.where(numpy.array(output_logs) == False)[0]
print("=======================================")
print("           Test summarys")
print("=======================================")
print("Tests are done at ",print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
print(f'There are total {Ntotal_tests} tests.')
print(f'There are {Npositive} successful tests.')
if Nnegative !=0 :
    print(f'There are {Nnegative} failed tests!')
    print('They are:')
    for i in idx:
        print('    ',test_name_list[i])
        