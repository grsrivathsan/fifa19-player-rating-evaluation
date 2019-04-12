
def loadFile(filename):
    with open(filename) as f:
        content = f.read().splitlines()

    output = content
    outlen = len(output);
    print(outlen)
    #print(output[0])
    #print((output[1:outlen]))
    #print((finalInput))
    return output[1:outlen],output[0];

def writeFile(output):
    with open('E:/Courses/Semester2/Data Mining/Project/Datasets/pyoutput.txt', 'w') as f:
        for item in output:
            f.write("%s\n" % item);

def skills_clean(lines,header):
    for line in lines:
        if(line.__contains__('+')):
            arr = list(line);
            outnum = int(arr[0]+arr[1])+int(arr[3]);
            output.append(outnum);
        elif line == '':
            output.append(0);
        else:
            output.append(int(line))

    print(output)
    #Write data to file
    #writeFile(output);
    sum = 0
    zcount = 0;
    for i in output:
        if i != 0:
            sum += i;

        else:
            zcount += 1;

    print(zcount)
    print(sum)
    print(len(output))
    nzcount = len(output) - zcount
    avg = sum / nzcount;
    print("Avg:",avg)
    #Missing values are replaced by mean

    finalOut = []
    finalOut.append(header);
    for i in output:
        if i != 0:
            finalOut.append(i);
        else:
            finalOut.append(int(avg));

    #Write final output to file
    writeFile(finalOut);
    print(len(finalOut))

if __name__ == '__main__':
    inputFile = 'E:/Courses/Semester2/Data Mining/Project/Datasets/pyinput.txt';
    output = []
    lines,header = loadFile(inputFile);
    print(header)
    print(lines)





