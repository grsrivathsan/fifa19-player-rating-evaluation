
def loadFile(filename):
    with open(filename) as f:
        content = f.read().splitlines()

    output = content
    outlen = len(output);
    #print(output[0])
    #print((output[1:outlen]))
    #print((finalInput))
    return output[1:outlen],output[0];

def normalizeWeight(lines):
    # Normalize Weight
    max = 0;
    sum = 0
    for i in lines:
        # print(int(wt)+5)
        if (i == ''):
            wt = 0
            output.append(wt)
        else:
            wt = int(i)
            if (wt > max):
                max = wt;
            output.append(wt)
        sum += wt;
    avg = "{0:.2f}".format(sum / len(output))
    # print(max, sum, avg)
    # print("lines len:", len(lines))
    # print("output len:", len(output))

    finalOut = []
    sum1 = 0
    for i in output:
        if (i == 0):
            finalOut.append(0);
            sum1 += 0
        else:
            num = (i / max) * 100;
            num1 = "{0:.2f}".format(num)
            finalOut.append(num1)
            sum1 += float(num1);

    #print("finalOut len:", len(finalOut))
    #print("finalOut:", finalOut);
    avg1 = "{0:.2f}".format(sum1 / len(finalOut))
    #print("finalOutAvg:", avg1)

    finalOut1 = []
    for i in finalOut:
        if (i == 0):
            finalOut1.append(avg1)
        else:
            finalOut1.append(i)

    print("finalOut1 len:", len(finalOut1))
    print(finalOut1)
    return finalOut1;

def writeFile(header,output):
    with open('E:/Courses/Semester2/Data Mining/Project/Datasets/pyoutput.txt', 'w') as f:
        f.write("%s\n" % header)
        for item in output:
            f.write("%s\n" % item);


if __name__ == '__main__':
    inputFile = 'E:/Courses/Semester2/Data Mining/Project/Datasets/pyinput.txt';
    output = []
    lines,header = loadFile(inputFile);
    print(lines,header)
    #Normalize weights
    finalOut = normalizeWeight(lines)

    #Write to file
    writeFile(header,finalOut);
