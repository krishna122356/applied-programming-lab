# First Assignment, roll no - EE19B147
import sys


if (len(sys.argv) != 2):  # Checks number of args
    print("Ensure a single filename is entered")
    exit()
arg0 = sys.argv[1]
if (arg0[-8:] != ".netlist"):  # Checks extension
    print("Incorrect file type")
    exit()
try:  # Checks if the file exists
    f = open(arg0)
    lines = f.readlines()
except:
    print("Unable to locate file")
    exit()
noOfLines = len(lines)
circuitDetect = False
endDetect = False
CollectionOfLines = []
lineCount = 0
# Extracts the segment from .circuit to .end. If either does not exist, provides error as such.
for i in range(noOfLines):
    if (circuitDetect == False):
        lineCount += 1
        if (lines[i][:8] == ".circuit"):
            circuitDetect = True
    else:
        if (lines[i][:4] != '.end'):
            CollectionOfLines.append(lines[i])
        else:
            endDetect = True
            break
if (circuitDetect == False) or (endDetect == False):
    print("Invalid Syntax,.circuit and/or .end not detected")
    exit()
printout = []  # Will contain all the strings that will be printed out.
for eachLine in CollectionOfLines:
    lineCount += 1
    CurrentLine = eachLine.split()
    Tokens = []
    for token in CurrentLine:
        if (token[0] == "#"):  # Detects comments, and if present removes them.
            break
        else:
            Tokens.append(token)
    if (Tokens == []):  # Checks next line, in case an empty line is inserted in between code.
        continue
    # Check token type, and check for the possible errors : Number of parameters, alphanumeric node names, etc. If
    # Syntactically correct, reverses the List of tokens and stores them in string ThisLine, which is then stored in
    # in the list printout.
    Element = Tokens[0]
    if (Element[0] in ["R", "L", "C", "V", "I"]):  #
        if (len(Tokens) != 4):
            print("Error at line", lineCount)
            print(
                "Incorrect number of parameters for the specified element. There should only be 3 additional parametes.")
            exit()
        if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False):
            print("Error at line", lineCount)
            print("Node names should be AlphaNumeric.")
            exit()
        ThisLine = ""
        Tokens.reverse()
        for token in Tokens:
            ThisLine += token + " "
        ThisLine = ThisLine[:-1]
        printout.append(ThisLine)
    elif (Element[0] in ["E", "G"]):
        if (len(Tokens) != 6):
            print("Error at line", lineCount)
            print(
                "Incorrect number of parameters for the specified element. There should only be 6 additional parametes.")
            exit()
        if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False) or (Tokens[3].isalnum() == False) or (
                Tokens[4].isalnum() == False):
            print("Error at line", lineCount)
            print("Node names should be AlphaNumeric.")
            exit()
        ThisLine = ""
        Tokens.reverse()
        for token in Tokens:
            ThisLine += token + " "
        ThisLine = ThisLine[:-1]
        printout.append(ThisLine)
    elif (Element[0] in ["H", "F"]):
        if (len(Tokens) != 5):
            print("Error at line", lineCount)
            print(
                "Incorrect number of parameters for the specified element. There should only be 5 additional parametes.")
            exit()
        if (Tokens[3][0] != "V"):
            print("Error at line", lineCount)
            print("Voltage name is not present, Syntax is incorrect (???)")
            exit()
        if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False):
            print("Error at line", lineCount)
            print("Node names should be AlphaNumeric.")
            exit()
        ThisLine = ""
        Tokens.reverse()
        for token in Tokens:
            ThisLine += token + " "
        ThisLine = ThisLine[:-1]
        printout.append(ThisLine)
    else:
        print("Error at line", lineCount)
        print("No such component.")
        exit()
# Reverse printout.
printout.reverse()
f.close()  # Close the file.
# Print the output.
for eachLine in printout:
    print(eachLine)

# Krishna Somasundaram, Roll no EE19B147




def DoDC(ListOfTokens):
    AllNodes={}


# Krishna Somasundaram, Roll no EE19B147
