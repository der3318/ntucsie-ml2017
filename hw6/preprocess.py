import numpy as np
np.random.seed(3318)  # for reproducibility

num = {"user": 0, "movie": 0, "rate0": 0, "rate1": 0, "rate2": 0, "rate3": 0, "rate4": 0, "rate5": 0,
    "age": 0, "occupation": 0}

# read training data
print("\x1b[0;31;40m<I/O> Reading Toekens in Training Data\x1b[0m")
with open("data/train.csv") as fin:
    for lineNum, line in enumerate(fin):
        if (lineNum + 1) % 3318 == 899874 % 3318:
            print("\rProcessing - Line " + str(lineNum + 1) + "/899874" , end = "")
        if lineNum == 0:    continue
        num["train"] = lineNum
        lineData = line.rstrip().split(",")
        num["user"] = max(num["user"], int(lineData[1]) + 1)
        num["movie"] = max(num["movie"], int(lineData[2]) + 1)
        num[ "rate" + lineData[3] ] += 1
print("\nnum[\"train\"] |", num["train"])
for i in range(6):
    print("num[\"rate" + str(i) + "\"] |", num["rate" + str(i)])

# read testing data
print("\x1b[0;31;40m<I/O> Reading Toekens in Testing Data\x1b[0m")
with open("data/test.csv") as fin:
    for lineNum, line in enumerate(fin):
        if (lineNum + 1) % 3318 == 100337 % 3318:
            print("\rProcessing - Line " + str(lineNum + 1) + "/100337" , end = "")
        if lineNum == 0:    continue
        num["test"] = lineNum
        lineData = line.rstrip().split(",")
        num["user"] = max(num["user"], int(lineData[1]) + 1)
        num["movie"] = max(num["movie"], int(lineData[2]) + 1)
print("\nnum[\"test\"] |", num["test"])
print("num[\"user\"] |", num["user"])
print("num[\"movie\"] |", num["movie"])

# read user data
print("\x1b[0;31;40m<I/O> Reading User Info\x1b[0m")
users = [ [] for i in range(num["user"]) ]
with open("data/users.csv") as fin:
    for lineNum, line in enumerate(fin):
        if (lineNum + 1) % 3318 == 6041 % 3318:
            print("\rProcessing - Line " + str(lineNum + 1) + "/6041" , end = "")
        if lineNum == 0:    continue
        lineData = line.rstrip().split(",")
        users[int(lineData[0])] = [lineData[1], int(lineData[2]), int(lineData[3])]
        num["age"] = max( num["age"], int(lineData[2]) )
        num["occupation"] = max(num["occupation"], int(lineData[3]) + 1)
print("\nnum[\"age\"] |", num["age"])
print("num[\"occupation\"] |", num["occupation"])
print("users[0] |", users[0])
print("users[1] |", users[1])

# parse user info
print("\x1b[0;31;40m<Parse> Building User Info Matrix\x1b[0m")
userMat = np.zeros( (num["user"], 2 + num["occupation"]) )
for userId in range(num["user"]):
    if len(users[userId]) == 0:  continue
    userMat[userId, 0] = 1. if users[userId][0] == "M" else 0
    userMat[userId, 1] = users[userId][1] / num["age"]
    userMat[userId, users[userId][2] + 2] = 1.
print("userMat.shape |", userMat.shape)
print("userMat[0, :] |", userMat[0, :])
print("userMat[1, :] |", userMat[1, :])

# read movie data
print("\x1b[0;31;40m<I/O> Reading Movie Info\x1b[0m")
movies = [ [] for i in range(num["movie"]) ]
(minYear, maxYear) = (1990, 1990)
tags = []
with open("data/movies.csv", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if (lineNum + 1) % 3318 == 3884 % 3318:
            print("\rProcessing - Line " + str(lineNum + 1) + "/3884" , end = "")
        if lineNum == 0:    continue
        lineData = line.rstrip().split(",")
        lineYear = float(lineData[-2].split("(")[-1].split(")")[0])
        if lineYear < minYear:  minYear = lineYear
        if lineYear > maxYear:  maxYear = lineYear
        movies[int(lineData[0])].append(lineYear)
        lineTags = lineData[-1].split("|")
        for lineTag in lineTags:
            if lineTag not in tags: tags.append(lineTag)
            movies[int(lineData[0])].append(lineTag)
tag2Idx = {}
for idx, tag in enumerate(tags):
    tag2Idx[tag] = (idx + 1)
num["tag"] = len(tags)
print("\nnum[\"tag\"] |", num["tag"])
print("movies[0] |", movies[0])
print("movies[1] |", movies[1])
print("minYear maxYear |", minYear, maxYear)
print("tags |", tags)
print("tag2Idx[\"Comedy\"] |", tag2Idx["Comedy"])

# parse movie info
movieMat = np.zeros( (num["movie"], 1 + num["tag"]) )
movieMat[:, 0] = np.random.rand(num["movie"])
for movieId in range(num["movie"]):
    if len(movies[movieId]) == 0:  continue
    movieMat[movieId, 0] = (movies[movieId][0] - minYear) / (maxYear - minYear)
    for tag in movies[movieId][1:]:
        movieMat[ movieId, tag2Idx[tag] ] = 1.
print("movieMat.shape |", movieMat.shape)
print("movieMat[0, :] |", movieMat[0, :])
print("movieMat[1, :] |", movieMat[1, :])

# save numbers
print("\x1b[0;31;40m<I/O> Numbers, Users and Movies Saved\x1b[0m")
np.save("model/num.npy", num)
np.save("model/userMat.npy", userMat)
np.save("model/movieMat.npy", movieMat)
np.save("model/tag2Idx.npy", tag2Idx)

