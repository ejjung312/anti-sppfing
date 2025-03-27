import os
import random
import shutil
from itertools import islice

outputFolderPath = "dataset/splitdata"
inputFolderPath = "dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
    print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)

# 디렉토리 생성
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)


# 이름 얻기
listNames = os.listdir(inputFolderPath)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split(".")[0])

uniqueNames = list(set(uniqueNames)) # 중복제거
print(uniqueNames)


# 섞기
random.shuffle(uniqueNames)


# 각 폴더마다 이미지 개수
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])


# 나머지 이미지는 train에
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining


# 리스트 나누기
lengthToSplit = [lenTrain, lenVal, lenTest]
input = iter(uniqueNames)
output = [list(islice(input, elem)) for elem in lengthToSplit]
print(f'Total Images: {lenData} \n Split: {len(output[0])} {len(output[1])} {len(output[2])}')


# 파일 복사
sequence = ['train', 'val', 'test']
for i, out in enumerate(output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')


print("Split Process Completed...")

# data.yaml 생성
dataYaml = f'path: ../data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")








